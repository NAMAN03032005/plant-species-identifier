# LeafScan — Plant Leaf Classification using Transfer Learning

**Educational Deep Learning Full-Stack Academic Project (TAE / Viva Demonstration)**

---

## 📌 Project Objective

LeafScan is a full-stack computer vision application built to identify **Plant Species and Leaf Condition (Healthy vs. Diseased status)** from leaf photographs using **Deep Learning** and **Transfer Learning**. Designed for college academic evaluation (TAE/viva), the project provides a real end-to-end Machine Learning pipeline: dataset curation, MobileNetV2 transfer learning, empirical performance auditing, Python Flask microservice integration, responsive Progressive Web App (PWA) frontend, native mobile camera/gallery triggers, and verified TensorFlow Lite mobile edge deployment.

---

## 🛠️ Technology Stack

| Layer | Technologies Used |
|:---|:---|
| **Frontend** | React 19, Vite 8, Lucide Icons, Vanilla CSS Design System |
| **PWA & Mobile** | Web App Manifest (`manifest.webmanifest`), Service Worker (`sw.js`), HTML5 Camera Capture |
| **Backend API** | Python 3.11, Flask 3.1, Flask-CORS, Werkzeug |
| **Deep Learning** | TensorFlow 2.21, Keras 3.15, MobileNetV2 (Pretrained on ImageNet) |
| **Mobile Model** | TensorFlow Lite (`.tflite` Standard & Float16 Quantized) |
| **Image Processing** | Pillow (PIL), NumPy, Scikit-learn, Matplotlib, Seaborn |

---

## 🏗️ System Architecture & Data Flow

```
User Photograph (React PWA UI / Mobile Camera)
       ↓
HTTP POST /api/predict (multipart/form-data)
       ↓
Python Flask Microservice (backend/app.py)
       ↓ (Format check & <= 5 MB size validation)
Pillow In-Memory Processing (Resize to 224x224 RGB Array)
       ↓
MobileNetV2 Transfer Learning Model (Loaded ONCE at server startup)
       ↓
Softmax Activation Layer (10 Species & Leaf Condition Classes)
       ↓
JSON Response (Top-1 Species, Confidence %, Top-3 Probabilities List)
       ↓
React Web Frontend (Parsed Species & Condition Cards, Top-3 Progress Bars)
```

---

## 📊 Plant Leaf Dataset (10 Categories)

The model is trained on **10 categories** curated from the academic **PlantVillage Dataset**:

| Index | Raw Model Label | Parsed Plant Species | Parsed Leaf Condition |
|:---:|:---|:---|:---|
| 0 | `Apple___Apple_scab` | Apple | Apple Scab (Diseased) |
| 1 | `Apple___Black_rot` | Apple | Black Rot (Diseased) |
| 2 | `Apple___healthy` | Apple | Healthy |
| 3 | `Cherry___healthy` | Cherry | Healthy |
| 4 | `Corn___Common_rust` | Corn (Maize) | Common Rust (Diseased) |
| 5 | `Corn___healthy` | Corn (Maize) | Healthy |
| 6 | `Grape___Black_rot` | Grape | Black Rot (Diseased) |
| 7 | `Grape___healthy` | Grape | Healthy |
| 8 | `Peach___Bacterial_spot` | Peach | Bacterial Spot (Diseased) |
| 9 | `Potato___Late_blight` | Potato | Late Blight (Diseased) |

- **Partition Split**: 70% Training / 15% Validation / 15% Test.
- **Data Integrity**: Verified 0 image hash duplicates across splits (`scripts/prepare_dataset.py`).

---

## 🧠 Model Architecture & Transfer Learning Strategy

- **Base Network**: MobileNetV2 (Pretrained on ImageNet, `include_top=False`).
- **Input Dimensions**: `224 x 224 x 3` (RGB).
- **Custom Classification Head**:
  `Input(224, 224, 3) -> DataAugmentation -> PreprocessInput -> MobileNetV2 Base -> GlobalAveragePooling2D -> Dropout(0.3) -> Dense(10, activation='softmax')`
- **Total Model Parameters**: `2,270,794`.
- **Training Strategy**:
  - **Phase 1 (Feature Extraction)**: Base layers frozen, Adam (`lr=1e-3`), 12 epochs.
  - **Phase 2 (Fine-Tuning)**: Top 30 layers unfrozen, Adam (`lr=1e-5`), 5 epochs.
- **Weight File Size**: `9.34 MB` ([leafscan_mobilenetv2.keras](file:///f:/DOC/My%20Projects/plant-species-identifier/model/leafscan_mobilenetv2.keras)).

---

## 📈 Measured Model Performance Metrics

Evaluated on **60 unseen test dataset images** ([model_evaluation_summary.md](file:///f:/DOC/My%20Projects/plant-species-identifier/model/model_evaluation_summary.md)):

| Metric | Measured Value | Description |
|:---|:---:|:---|
| **Test Loss** | `1.4701` | Categorical cross-entropy loss |
| **Test Accuracy** | **`50.00%`** | Measured accuracy across unseen test images |
| **Weighted Precision** | `0.5656` | Weighted macro precision |
| **Weighted Recall** | `0.5000` | Weighted macro recall |
| **Weighted F1-Score** | `0.4623` | Weighted macro F1-score |

---

## 📱 TensorFlow Lite Mobile Deployment

The trained model was exported to TensorFlow Lite format using `scripts/convert_to_tflite.py`:

| Model Format | Disk Size | Size Reduction | Test Accuracy | Prediction Parity Match Rate |
|:---|:---:|:---:|:---:|:---:|
| **Keras Baseline (`leafscan_mobilenetv2.keras`)** | `9.34 MB` | `0.00%` | `50.00%` | Baseline (`10/10`) |
| **Standard TFLite (`leafscan_mobilenetv2.tflite`)** | **`8.53 MB`** | `8.70%` | **`46.67%`** | **`10 / 10 (100.0% Match)`** |
| **Float16 TFLite (`leafscan_mobilenetv2_float16.tflite`)** | **`4.33 MB`** | **`53.66%`** | `35.00%` | `6 / 10 (60.0% Match)` |

> **Architectural Note**:
> - **Current PWA Web Inference**: `React PWA -> Flask REST API -> MobileNetV2 Keras Model`.
> - **Prepared Mobile Model**: `leafscan_mobilenetv2.tflite` is ready for native Android (Kotlin) or iOS (Swift) embedding.

---

## 📱 Mobile-First PWA Experience & Features

- **📷 Mobile Camera Capture**: HTML5 `accept="image/*" capture="environment"` requesting rear camera.
- **🖼️ Photo Gallery Support**: Image picker filtering JPG, PNG, and WEBP files.
- **📱 PWA Installation**: Web App Manifest (`manifest.webmanifest`), 192x192 & 512x512 icons, and interactive install prompt banner (`InstallPrompt.jsx`).
- **⚡ Service Worker Caching**: Static app shell caching (`sw.js`) with network-first bypass for `/api/*`.
- **🍔 Mobile Navigation**: Touch-friendly hamburger drawer menu auto-closing upon section selection.
- **💡 Quality Guidance Card**: "Tips for better results" helping users take optimal leaf photographs.

---

## 🚀 How to Run Locally

### 1. Backend Setup (Flask API)

```bash
# Navigate to backend directory
cd backend

# Activate virtual environment
venv\Scripts\activate

# Start Flask REST API server
python app.py
```
> Flask API runs at `http://127.0.0.1:5000`  
> Health check: `http://127.0.0.1:5000/api/health`

### 2. Frontend Setup (React + Vite PWA)

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies & start Vite dev server
npm install
npm run dev
```
> Vite dev server runs at `http://localhost:5173`

---

## 🧪 Testing & Verification Scripts

```bash
# 1. Run complete End-to-End Automated Test Suite (Health, 5 Images, Security Rejection, TFLite):
python scripts/end_to_end_test.py

# 2. Run TFLite Parity Verification:
python scripts/test_tflite.py

# 3. Test Standalone Mobile TFLite CLI Inference:
python scripts/mobile_inference_demo.py dataset/test/Potato___Late_blight/Potato___Late_blight_sample_007.jpg

# 4. Verify React Frontend Production Build:
cd frontend && npm run build
```

---

## 📚 Project Documentation Directory (`docs/`)

- [system_architecture.md](file:///f:/DOC/My%20Projects/plant-species-identifier/docs/system_architecture.md): Full-stack system design & data flow diagrams.
- [api_documentation.md](file:///f:/DOC/My%20Projects/plant-species-identifier/docs/api_documentation.md): REST API endpoints specification and status codes.
- [testing_report.md](file:///f:/DOC/My%20Projects/plant-species-identifier/docs/testing_report.md): Structured test matrix & verification results.
- [model_explanation.md](file:///f:/DOC/My%20Projects/plant-species-identifier/docs/model_explanation.md): Academic viva Q&A guide for Transfer Learning & MobileNetV2.
- [mobile_deployment.md](file:///f:/DOC/My%20Projects/plant-species-identifier/docs/mobile_deployment.md): PWA web deployment vs. native mobile TFLite integration.

---

## ⚠️ Academic Limitations & Disclaimers

1. **Category Boundary**: LeafScan classifies categories represented in its 10 trained classes.
2. **Softmax Confidence**: Confidence percentage reflects the model's Softmax probability distribution for trained categories. It does not guarantee prediction correctness on out-of-distribution photos.
3. **Academic Purpose**: Built specifically for college Machine Learning / Deep Learning TAE evaluation and viva defense.
