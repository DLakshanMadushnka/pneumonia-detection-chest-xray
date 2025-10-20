import gradio as gr
import tensorflow as tf
import numpy as np
import cv2
import os
from utils import generate_gradcam, overlay_heatmap
from config import Config

# Load model
model = tf.keras.models.load_model(Config.MODEL_PATH, custom_objects={'focal_loss_fixed': focal_loss()})

def preprocess_image(image):
    """Preprocess input image."""
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    img = cv2.resize(image, Config.IMG_SIZE)
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, 0)

def classify_pneumonia(image):
    """Classify and visualize."""
    if image is None:
        return "Please upload an image.", None
    
    img_array = preprocess_image(image)
    prediction = model.predict(img_array)[0][0]
    label = "Pneumonia" if prediction > 0.5 else "Normal"
    confidence = f"{max(prediction, 1 - prediction) * 100:.2f}%"
    
    # Grad-CAM
    heatmap = generate_gradcam(model, img_array)
    superimposed = overlay_heatmap(heatmap, image)
    
    # Save result
    plt.figure(figsize=(6, 6))
    plt.imshow(superimposed)
    plt.title(f"Predicted: {label} (Confidence: {confidence})")
    plt.axis('off')
    plt.savefig(os.path.join(Config.OUTPUT_PATH, 'result.png'), bbox_inches='tight')
    plt.close()
    
    return f"Predicted: {label}\nConfidence: {confidence}", os.path.join(Config.OUTPUT_PATH, 'result.png')

# Gradio Interface (Enhanced UI)
with gr.Blocks(
    title="Pneumonia Detection",
    theme=gr.themes.Soft(),
    css="""
    body {background-color: #f0f8ff;}
    .gradio-container {max-width: 800px; margin: auto;}
    """
) as demo:
    gr.Markdown("# Advanced Pneumonia Detection from Chest X-Rays")
    gr.Markdown("Upload a chest X-ray for classification with Grad-CAM visualization.")
    
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(type="numpy", label="Upload X-Ray Image")
            btn_classify = gr.Button("Classify Image", variant="primary")
            btn_reset = gr.Button("Reset")
        
        with gr.Column():
            output_text = gr.Textbox(label="Prediction", interactive=False)
            output_img = gr.Image(label="Grad-CAM Heatmap", interactive=False)
    
    # Events
    btn_classify.click(classify_pneumonia, inputs=input_img, outputs=[output_text, output_img])
    btn_reset.click(lambda: (None, None), outputs=[input_img, output_img])
    
    # Examples (Optional: Add sample images)
    gr.Examples(
        examples=[["path/to/sample_normal.jpg"], ["path/to/sample_pneumonia.jpg"]],
        inputs=input_img
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
