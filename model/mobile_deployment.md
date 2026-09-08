# LeafScan — Mobile Deployment Architecture & TensorFlow Lite Documentation

**Educational Deep Learning Academic Project (TAE / Viva Demonstration)**

---

## 📌 Executive Summary

TensorFlow Lite (TFLite) is an open-source, cross-platform deep learning framework designed by Google for executing machine learning models on resource-constrained mobile and edge devices (iOS, Android, Raspberry Pi, Microcontrollers).

By converting our trained **MobileNetV2** Keras model (`leafscan_mobilenetv2.keras`) into **TensorFlow Lite format** (`leafscan_mobilenetv2_float16.tflite`), we enable **on-device computer vision inference** without requiring an active web server or internet connection.

---

## 🏗️ Architectural Workflows Comparison

### 1. Web Application Inference Workflow (Current Production Setup)

```
User Photograph (React UI)
       ↓
HTTP POST Request (/api/predict)
       ↓
Python Flask REST Backend
       ↓
Full Keras Model (leafscan_mobilenetv2.keras)
       ↓
Prediction Response (JSON) -> Displayed in React UI
```

### 2. On-Device Mobile Inference Workflow (TensorFlow Lite Setup)

```
User Photograph (Mobile App / Camera)
       ↓
On-Device Preprocessing (224x224 RGB Array)
       ↓
TensorFlow Lite Interpreter
       ↓
Quantized TFLite Model (leafscan_mobilenetv2_float16.tflite)
       ↓
Immediate Local Prediction (Native UI / Android / iOS)
```

---

## 💡 Key Architectural Distinctions

| Feature / Metric | Web Application Architecture | Mobile TFLite Architecture |
|:---|:---|:---|
| **Execution Environment** | Server-side (Python Flask microservice) | Client-side (Mobile CPU / GPU / NPU) |
| **Model Weight File** | `leafscan_mobilenetv2.keras` | `leafscan_mobilenetv2_float16.tflite` |
| **Network Dependency** | Mandatory (Requires HTTP backend server) | Zero (Works 100% Offline) |
| **Inference Latency** | ~200ms – 1000ms (Network Round-Trip) | ~15ms – 45ms (Native Hardware Acceleration) |
| **Privacy & Security** | Image stream transmitted over HTTP | Image stream processed 100% locally on device |
| **Server Bandwidth Cost** | Proportional to daily active users | Zero server bandwidth cost |

---

## ⚡ Benefits of TensorFlow Lite for Mobile Deployment

1. **Lightweight Model Footprint**:
   Float16 quantization converts 32-bit floating-point weights into 16-bit half-precision floats, reducing disk storage and RAM consumption by ~50% with near-zero loss in prediction accuracy.

2. **Hardware Acceleration**:
   TensorFlow Lite utilizes Android Neural Networks API (NNAPI), Apple Metal API, and GPU delegates for real-time inference on mobile devices.

3. **Offline Plant Identification**:
   Farmers and field researchers often work in agricultural environments with weak or no cellular connection. On-device TFLite allows leaf diagnosis directly in remote locations.

4. **Privacy Preservation**:
   Private agricultural photos never need to leave the user's device, adhering to data privacy and regulatory compliance.

---

## 📱 Mobile Deployment Strategy

### CURRENT MOBILE DEPLOYMENT (Step 9)

The LeafScan web application is fully mobile-responsive and installable as a **Progressive Web App (PWA)**:
- **Mobile Camera Support**: Directly invokes phone rear camera (`capture="environment"`) or opens the photo gallery.
- **Installability**: Includes Web App Manifest (`manifest.webmanifest`), icons (192x192, 512x512), and Service Worker (`sw.js`).
- **Web Inference Execution**: The current PWA sends leaf images to the Flask REST API (`POST /api/predict`), which executes the **MobileNetV2 Keras model** (`leafscan_mobilenetv2.keras`).

### FUTURE ON-DEVICE DEPLOYMENT

The TensorFlow Lite model weight files (`leafscan_mobilenetv2.tflite` & `leafscan_mobilenetv2_float16.tflite`) are generated and verified for future on-device integration:
- **Native Android / iOS Application**: A native mobile client (Kotlin, Swift, Flutter, or React Native) can embed `.tflite` weights and run predictions locally on the phone's NPU/GPU without a server.
- **Offline In-Browser TFLite**: Browser-side execution via WebAssembly (`tfjs-tflite`) can be integrated in future steps.
