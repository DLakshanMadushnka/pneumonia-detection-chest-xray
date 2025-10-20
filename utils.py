import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
import shap
import cv2
import matplotlib.pyplot as plt

def focal_loss(gamma=2.0, alpha=0.25):
    """Focal loss for imbalanced classification."""
    def focal_loss_fixed(y_true, y_pred):
        pt = y_pred * y_true + (1 - y_pred) * (1 - y_true)
        pt = K.clip(pt, K.epsilon(), 1 - K.epsilon())
        return -K.sum(alpha * K.pow(1. - pt, gamma) * K.log(pt))
    return focal_loss_fixed

def generate_gradcam(model, img_array, layer_name='block5_conv3'):
    """Generate Grad-CAM heatmap."""
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = K.mean(tf.multiply(tf.expand_dims(pooled_grads, axis=0), conv_outputs), axis=-1)
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    return heatmap

def overlay_heatmap(heatmap, image):
    """Overlay heatmap on image."""
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed = heatmap * 0.4 + image * 255
    return np.clip(superimposed, 0, 255).astype(np.uint8)

def compute_metrics(y_true, y_pred_proba, y_pred):
    """Compute AUC, F1, etc."""
    auc = roc_auc_score(y_true, y_pred_proba)
    f1 = f1_score(y_true, y_pred)
    return {'AUC': auc, 'F1': f1}
