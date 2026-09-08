# LeafScan — Deep Learning Model Explanation (Viva Defense Guide)

**Academic TAE Presentation & Viva Technical Q&A Guide**

---

## 📚 Core Deep Learning Concepts

### 1. What is a Convolutional Neural Network (CNN)?
A **Convolutional Neural Network (CNN)** is a specialized deep learning architecture designed for visual image processing. Unlike traditional dense neural networks, CNNs use **kernel convolutions** to extract spatial features (edges, textures, shapes, leaf vein patterns) directly from raw image pixels without manual feature engineering.

---

### 2. What is Transfer Learning?
**Transfer Learning** is a machine learning technique where a model trained on a large dataset (e.g., ImageNet with 1.4 million general images) is repurposed as the starting point for a secondary specialized task (e.g., plant leaf classification).

**Why use Transfer Learning in LeafScan?**
- **Data Efficiency**: Eliminates the need for hundreds of thousands of leaf training images.
- **Fast Training**: The feature extraction layers are already trained to recognize universal shapes and visual structures.
- **Higher Accuracy**: Prevents overfitting on smaller custom datasets.

---

### 3. What is MobileNetV2?
**MobileNetV2** is a lightweight deep neural network architecture designed by Google specifically for mobile, web, and embedded vision applications.

**Key Technical Features**:
- **Inverted Residual Blocks**: Reduces memory footprint by expanding channels temporarily inside block shortcuts.
- **Depthwise Separable Convolutions**: Splits standard convolutions into a depthwise spatial filter and a pointwise 1x1 channel projection. This reduces mathematical parameters by ~8-9x compared to standard CNNs.

---

### 4. What is ImageNet?
**ImageNet** is a famous benchmark dataset containing over 14 million annotated photographs spanning 1,000 general categories (animals, vehicles, objects). Pretraining MobileNetV2 on ImageNet allows the model's lower convolutional layers to act as powerful general feature detectors.

---

### 5. What is Fine-Tuning?
Fine-tuning is the process of unfreezing top convolutional layers of a pretrained base model and training them on a custom dataset with a very low learning rate (e.g. `1e-5`). In LeafScan, Phase 1 freezes MobileNetV2 to train top classification layers, while Phase 2 unfreezes the top 30 layers for fine-tuning.

---

### 6. What is Data Augmentation?
**Data Augmentation** applies random spatial transformations (horizontal flips, slight rotations up to 10%, zooms up to 10%, translations) to training images on the fly. This prevents memorization (overfitting) and ensures the model generalizes well to real-world leaf photos taken at different angles or lighting conditions.

---

### 7. What is Softmax Activation?
The **Softmax** function is applied to the final output layer of the neural network. It transforms raw numeric output logits into a normalized probability distribution where all class probabilities sum to exactly 1.0 (100%).

$$\text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$$

---

### 8. What does Confidence Percentage represent?
The confidence score (e.g. `38.07%`) represents the model's Softmax probability distribution for trained categories. It indicates relative certainty among the 10 trained species classes, but does not guarantee correctness if the photo contains background artifacts or unrepresented species.

---

### 9. What is TensorFlow Lite (TFLite)?
**TensorFlow Lite** is Google's lightweight runtime framework for executing trained neural networks on mobile devices (Android, iOS) and edge hardware. It converts `.keras` models into compact `.tflite` flatbuffers, applying Float16 quantization to shrink model size from `9.34 MB` to `4.33 MB`.

---

### 10. Why is MobileNetV2 suitable for Mobile Deployment?
MobileNetV2 has only **2.27 million parameters** (compared to ResNet50 with 25 million parameters). Its low parameter count and depthwise separable convolutions make it lightweight enough to run predictions in milliseconds directly on a smartphone CPU or NPU without draining battery or requiring a dedicated GPU.
