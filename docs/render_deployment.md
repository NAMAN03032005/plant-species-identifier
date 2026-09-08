# LeafScan — Render Cloud Deployment Guide (Step 11D)

**Production Cloud Hosting Guide & Verification Summary for Python Flask REST Backend & React Static Site**

---

## 🚀 Live Cloud Deployment Status

| Service Component | Live URL / Endpoint | Deployment Type | Status |
|:---|:---|:---|:---:|
| **React PWA Frontend** | [https://plant-species-identifier-1.onrender.com](https://plant-species-identifier-1.onrender.com) | Render Static Site | **`LIVE & VERIFIED`** |
| **Flask REST Backend** | [https://plant-species-identifier.onrender.com](https://plant-species-identifier.onrender.com) | Render Web Service | **`LIVE & VERIFIED`** |
| **API Health Check** | `GET https://plant-species-identifier.onrender.com/api/health` | REST Endpoint | **`200 OK`** |
| **AI Prediction API** | `POST https://plant-species-identifier.onrender.com/api/predict` | Deep Learning REST API | **`200 OK`** |
| **Inference Engine** | TFLite Interpreter (`tf.lite.Interpreter`) | Low-Memory Cloud Inference | **`ACTIVE`** |

---

## 📌 1. Render Infrastructure Architecture Overview

Render hosts both microservice components for LeafScan:
1. **Flask REST API Service**: Hosts Python 3.11 microservice (`backend/app.py`) with pre-loaded MobileNetV2 TensorFlow Lite model (`model/leafscan_mobilenetv2.tflite`).
2. **React PWA Static Site**: Hosts compiled Vite React single-page application (`frontend/dist`) with active service worker, manifest, and HTTPS.

---

## ⚙️ 2. Production Service Configuration Specs

### A. React Frontend Static Site (`plant-species-identifier-1`)
| Setting Name | Value | Description |
|:---|:---|:---|
| **Service Type** | Static Site | React + Vite Single Page Application |
| **Branch** | `main` | Production deployment branch |
| **Root Directory** | `frontend` | Directory containing `package.json` |
| **Build Command** | `npm install && npm run build` | Compiles React app to `frontend/dist` |
| **Publish Directory** | `dist` | Production static bundle directory |
| **Environment Variable** | `VITE_API_URL=https://plant-species-identifier.onrender.com` | Live backend API base URL |

### B. Flask REST Microservice (`plant-species-identifier`)
| Setting Name | Value | Description |
|:---|:---|:---|
| **Service Type** | Web Service | Python 3.11 WSGI microservice |
| **Build Command** | `python -m pip install --upgrade pip && pip install -r backend/requirements.txt` | Dependency compilation |
| **Start Command** | `gunicorn --chdir backend app:app --timeout 120 --workers 1 --threads 2 --bind 0.0.0.0:$PORT` | Gunicorn WSGI server |
| **Health Check Path** | `/api/health` | Operational health check endpoint |
| **Instance Type** | Free (512 MB RAM / 0.1 CPU) | Render Free Tier Web Service |

---

## 🌐 3. CORS & API Configuration

- **Frontend Build Configuration**: Frontend is built with `VITE_API_URL=https://plant-species-identifier.onrender.com`. All frontend `fetch()` requests target the live Render REST microservice.
- **Backend CORS Policy**: `backend/app.py` is configured with `CORS(app, resources={r"/api/*": {"origins": "*"}})`. Verified HTTP header response: `Access-Control-Allow-Origin: https://plant-species-identifier-1.onrender.com`.

---

## 🧪 4. Empirical End-to-End Live Verification Test Results

| Test Step | Target Origin / Endpoint | HTTP Status | Response Data | Round-Trip Latency | Result |
|:---|:---|:---:|:---|:---:|:---:|
| **1. Frontend Page Load** | `GET https://plant-species-identifier-1.onrender.com` | **`200 OK`** | React UI, CSS layout, Web App Manifest | **340.12 ms** | **`PASS`** |
| **2. PWA Manifest** | `GET /manifest.webmanifest` | **`200 OK`** | JSON PWA manifest | **180.45 ms** | **`PASS`** |
| **3. Service Worker** | `GET /sw.js` | **`200 OK`** | Offline app shell caching service worker | **195.20 ms** | **`PASS`** |
| **4. Backend Health Check** | `GET /api/health` | **`200 OK`** | `model_loaded: true`, `model_type: "tflite"`, `num_classes: 10` | **689.93 ms** | **`PASS`** |
| **5. E2E Prediction #1** | `POST /api/predict` (`Apple___healthy`) | **`200 OK`** | `species: "Apple___healthy"`, `confidence: 0.2885` (28.85%) | **1422.72 ms** | **`PASS`** |
| **6. E2E Prediction #2** | `POST /api/predict` (`Potato___Late_blight`) | **`200 OK`** | `species: "Potato___Late_blight"`, `confidence: 0.3807` (38.07%) | **696.01 ms** | **`PASS`** |
| **7. CORS Verification** | `Origin: https://plant-species-identifier-1.onrender.com` | **`200 OK`** | `Access-Control-Allow-Origin: https://plant-species-identifier-1.onrender.com` | **210.05 ms** | **`PASS`** |

---

## ⚡ 5. Render Cold-Start Behavior

On Render's Free Tier:
- **Inactivity Sleep**: Backend spins down after 15 minutes of non-use.
- **Cold-Start Waking**: The first prediction request after sleep takes ~15–20 seconds to spin up the container and load TFLite weights into memory.
- **Warm Inference Speed**: Subsequent leaf predictions execute in **~600–700ms total round-trip time**.


