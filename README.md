# Pneumonia Detection from Chest X-Rays

A fine-tuned VGG16 model for binary classification of chest X-rays (Normal vs. Pneumonia), incorporating focal loss, data augmentation, and Grad-CAM for explainability. Deployed via Gradio for interactive use.

## Setup
1. Clone: `git clone https://github.com/DLakshanMadushnka/pneumonia-detection-chest-xray.git`
2. Install: `pip install -r requirements.txt`
3. Download data: Place Kaggle `kaggle.json` and run `python download_data.py`.
4. Train: `python train.py`
5. Deploy: `python app.py`

## Features
- **Model**: VGG16 with focal loss (AUC > 0.95 expected).
- **Evaluation**: ROC-AUC, F1-score, confusion matrix.
- **Interpretability**: Grad-CAM heatmaps.
- **Deployment**: Gradio UI; Docker-ready.

## Results
- Test Accuracy: ~92% (dataset-dependent).
- See `outputs/` for metrics and visuals.

## License
MIT
