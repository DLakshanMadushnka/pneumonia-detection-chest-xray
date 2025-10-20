import os
import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler  # For MixUp-like balancing
from utils import focal_loss, compute_metrics
from config import Config

# Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
tf.random.set_seed(Config.SEED)
np.random.seed(Config.SEED)

def load_data():
    """Load with augmentation and balancing."""
    os.makedirs(Config.OUTPUT_PATH, exist_ok=True)
    
    train_datagen = ImageDataGenerator(
        rescale=1./255, rotation_range=20, width_shift_range=0.2,
        height_shift_range=0.2, shear_range=0.2, zoom_range=0.2,
        horizontal_flip=True, fill_mode='nearest', validation_split=0.2
    )
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    train_gen = train_datagen.flow_from_directory(
        Config.TRAIN_DIR, target_size=Config.IMG_SIZE, batch_size=Config.BATCH_SIZE,
        class_mode='binary', subset='training', seed=Config.SEED
    )
    val_gen = train_datagen.flow_from_directory(
        Config.TRAIN_DIR, target_size=Config.IMG_SIZE, batch_size=Config.BATCH_SIZE,
        class_mode='binary', subset='validation', seed=Config.SEED
    )
    test_gen = test_datagen.flow_from_directory(
        Config.TEST_DIR, target_size=Config.IMG_SIZE, batch_size=Config.BATCH_SIZE,
        class_mode='binary', shuffle=False
    )
    
    # Class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(train_gen.classes), y=train_gen.classes)
    Config.CLASS_WEIGHTS = dict(enumerate(class_weights))
    
    return train_gen, val_gen, test_gen

def build_model():
    """Enhanced VGG16 with focal loss and regularization."""
    base = VGG16(weights='imagenet', include_top=False, input_shape=(*Config.IMG_SIZE, 3))
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)  # Added dropout
    predictions = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base.input, outputs=predictions)
    
    # Fine-tuning
    for layer in base.layers[:-4]:
        layer.trainable = False
    for layer in base.layers[-4:]:
        layer.trainable = True
    
    model.compile(
        optimizer=Adam(learning_rate=Config.LEARNING_RATE),
        loss=focal_loss(gamma=Config.GAMMA, alpha=Config.FOCAL_ALPHA),
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    return model

def train_and_evaluate():
    """Full training pipeline."""
    train_gen, val_gen, test_gen = load_data()
    model = build_model()
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=3),
        tf.keras.callbacks.ModelCheckpoint(Config.MODEL_PATH, save_best_only=True)
    ]
    
    history = model.fit(
        train_gen, epochs=Config.EPOCHS, validation_data=val_gen,
        class_weight=Config.CLASS_WEIGHTS, callbacks=callbacks
    )
    
    # Evaluation
    y_pred_proba = model.predict(test_gen)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    y_true = test_gen.classes
    metrics = compute_metrics(y_true, y_pred_proba, y_pred)
    logger.info(f"Metrics: {metrics}")
    
    # Plots
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    plt.plot(fpr, tpr, label=f'ROC (AUC={metrics["AUC"]:.3f})')
    plt.legend()
    plt.savefig(os.path.join(Config.OUTPUT_PATH, 'training_eval.png'))
    plt.close()
    
    model.save(Config.MODEL_PATH)
    logger.info("Training complete.")

if __name__ == "__main__":
    train_and_evaluate()
