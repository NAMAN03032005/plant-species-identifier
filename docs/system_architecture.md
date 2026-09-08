# LeafScan — Full-Stack System Architecture & Data Flow

**Educational Deep Learning Academic Project (TAE / Viva Defense)**

---

## 📌 1. High-Level Architecture Overview

LeafScan is designed as a decoupled, multi-tier computer vision system separating frontend client presentation, backend microservices, deep learning inference engines, and mobile edge runtime.

```
+-----------------------------------------------------------------------+
|                            USER CLIENT                                |
|  Mobile Camera / Photo Gallery / Desktop Drag & Drop (React PWA)      |
+-----------------------------------------------------------------------+
                                   |
                HTTP POST /api/predict (multipart/form-data)
                                   v
+-----------------------------------------------------------------------+
|                        FLASK REST MICROSERVICE                        |
|  - File format validation (JPG/PNG/WEBP) & Max 5 MB size check        |
|  - Pillow In-Memory Image Preprocessing (224x224 RGB Array)           |
+-----------------------------------------------------------------------+
                                   |
                                   v
+-----------------------------------------------------------------------+
|                    DEEP LEARNING INFERENCE ENGINE                     |
|  - MobileNetV2 Transfer Learning Model (leafscan_mobilenetv2.keras)   |
|  - Pre-loaded ONCE at Flask server startup                            |
|  - Top Softmax Layer (10 Species & Condition Classes)                 |
+-----------------------------------------------------------------------+
                                   |
                     JSON Response (Top-1 & Top-3)
                                   v
+-----------------------------------------------------------------------+
|                     REACT PREDICTION RESULT UI                        |
|  - Parsed Plant Species & Leaf Condition Display Cards                |
|  - Accessible ARIA Confidence Progress Bars                           |
|  - Top-3 Candidate Probabilities List                                 |
+-----------------------------------------------------------------------+
```

---

## 🏗️ 2. Architectural Components

### A. Frontend Layer (React + Vite PWA)
- **Framework**: React 19 + Vite 8
- **PWA Features**: Web App Manifest (`manifest.webmanifest`), Service Worker (`sw.js`) static asset caching, Install Prompt banner (`InstallPrompt.jsx`).
- **Input Triggers**: Native mobile camera capture (`capture="environment"`), gallery picker, desktop drag-and-drop.
- **State Management**: Reactive component state tracking upload, image preview, prediction results, and session analysis history.

### B. Backend Layer (Python Flask Microservice)
- **Framework**: Flask 3.1 + Flask-CORS + Werkzeug
- **Endpoints**:
  - `GET /api/health`: Operational status & model loading check.
  - `POST /api/upload`: Image format and file size validation (< 5 MB).
  - `POST /api/predict`: Real MobileNetV2 model inference.
- **Pre-loading Protocol**: Model weights (`leafscan_mobilenetv2.keras`) and species class labels (`class_names.json`) are loaded **ONCE** at Flask initialization.

### C. Deep Learning Layer (TensorFlow / Keras MobileNetV2)
- **Base Model**: MobileNetV2 (ImageNet pretrained, `include_top=False`).
- **Input Shape**: `(224, 224, 3)` RGB float32.
- **Custom Top Head**: `GlobalAveragePooling2D -> Dropout(0.3) -> Dense(10, activation='softmax')`.
- **Pre-processing**: `tf.keras.applications.mobilenet_v2.preprocess_input` embedded in graph.

### D. Mobile Edge Layer (TensorFlow Lite)
- **Standard TFLite Model**: `model/leafscan_mobilenetv2.tflite` (`8.53 MB`, 100% prediction parity with Keras).
- **Float16 Quantized TFLite Model**: `model/leafscan_mobilenetv2_float16.tflite` (`4.33 MB`, 53.66% smaller footprint).
- **Standalone CLI Inference**: `scripts/mobile_inference_demo.py`.
