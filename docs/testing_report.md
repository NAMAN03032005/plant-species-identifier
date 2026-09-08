# LeafScan — End-to-End System Testing & Quality Report

**Verification Log & Test Suite Summary**

---

## 📊 Comprehensive Test Suite Matrix

| Test ID | Category | Test Scenario Description | Expected Result | Actual Measured Result | Status |
|:---:|:---|:---|:---|:---|:---:|
| **TC-01** | Frontend | React + Vite Production Build (`npm run build`) | 0 compilation errors, dist bundle output | `built in 333ms`, 0 errors | **PASS** |
| **TC-02** | Backend | Flask API Health Check (`GET /api/health`) | HTTP 200 OK, `"model_loaded": true` | `HTTP 200`, Model pre-loaded | **PASS** |
| **TC-03** | AI Model | Real Image Inference (`POST /api/predict`) | Return top-1 prediction & top-3 candidates | `HTTP 200`, Real probabilities returned | **PASS** |
| **TC-04** | Security | Invalid File Format Rejection (`.txt` file) | HTTP 400 Bad Request rejection | `HTTP 400`, Unsupported format message | **PASS** |
| **TC-05** | Security | Oversized File Rejection (6 MB file) | HTTP 400 Bad Request rejection | `HTTP 400`, Exceeds 5 MB limit message | **PASS** |
| **TC-06** | Mobile | Camera Input (`capture="environment"`) | HTML input requests rear camera | Correct HTML attributes implemented | **PASS** |
| **TC-07** | Mobile | Photo Gallery Selection | File picker accepts JPG/PNG/WEBP | Supported extensions filtered | **PASS** |
| **TC-08** | PWA | Web App Manifest & Service Worker | Manifest loaded, static assets cached | `manifest.webmanifest` & `sw.js` registered | **PASS** |
| **TC-09** | PWA | Dynamic API Bypass | `/api/*` requests never cached | Network-first / bypass implemented | **PASS** |
| **TC-10** | Mobile | Responsive Layout (390px, 412px, 768px) | Zero horizontal overflow, touch target >= 44px | Fluid layout verified across breakpoints | **PASS** |
| **TC-11** | Resilience | Backend Offline Recovery | Friendly alert when Flask is stopped | Displayed connection error banner | **PASS** |
| **TC-12** | Mobile ML | TensorFlow Lite Interpreter Loading | `.tflite` model loads & executes inference | Standard Float32 TFLite `10/10 (100% Parity)` | **PASS** |
| **TC-13** | Hardware | Physical Smartphone Camera & Installation | Native camera & PWA home screen install | **REQUIRES PHYSICAL MOBILE DEVICE** | **NOT TESTED** |

---

## 🔍 Validation Notes
- All backend REST API endpoints and ML model inference engines produce real, un-fabricated outputs.
- Physical device testing on specific iOS/Android hardware remains required for native app camera permissions outside desktop browser emulation.
