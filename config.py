import os

class Config:
    # Paths
    DATA_PATH = '/content/drive/MyDrive/chest_xray_dataset/chest_xray/'  # Post-unzip
    TRAIN_DIR = os.path.join(DATA_PATH, 'train')
    VAL_DIR = os.path.join(DATA_PATH, 'val')  # Use train split for val
    TEST_DIR = os.path.join(DATA_PATH, 'test')
    MODEL_PATH = os.path.join(DATA_PATH, 'pneumonia_vgg16_finetuned.keras')
    OUTPUT_PATH = os.path.join(DATA_PATH, 'outputs/')
    
    # Data/Model
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32
    EPOCHS = 20  # Increased for convergence
    LEARNING_RATE = 1e-5
    CLASS_WEIGHTS = None  # Computed dynamically
    FOCAL_ALPHA = 0.25  # For focal loss
    GAMMA = 2.0  # Focal gamma
    
    # Reproducibility
    SEED = 42
