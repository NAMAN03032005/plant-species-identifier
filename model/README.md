# Model Directory

This directory will store trained Deep Learning models, weight files, and class label metadata for the **LeafScan** application.

## Planned Model Architecture & Transfer Learning
- **Base Architecture**: MobileNetV2 (Pre-trained on ImageNet)
- **Framework**: TensorFlow 2.x / Keras
- **Input Shape**: `(224, 224, 3)`
- **Output Layer**: Softmax activation with target plant species classes
- **Saved Model Format**: `.keras` or `.h5` format

## Future Files in this Folder (Step 2+)
- `leafscan_mobilenetv2.keras`: Saved fine-tuned model weights.
- `class_indices.json`: JSON dictionary mapping numerical class indices to plant species names (e.g., `{"0": "Apple___Apple_scab", "1": "Corn___Common_rust"}`).
- `train_model.py`: Script to train and evaluate the model using Transfer Learning.

> **Note**: The actual model file will be trained and integrated in a subsequent development step.
