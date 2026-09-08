# LeafScan — Render Web Service Deployment Guide (Step 11C)

**Production Cloud Hosting Guide & Verification Summary for Python Flask REST Microservice**

---

## 🚀 Live Render Backend Status

| Metric / Setting | Deployed Value |
|:---|:---|
| **Live Backend Service URL** | [https://plant-species-identifier.onrender.com](https://plant-species-identifier.onrender.com) |
| **Health Check Endpoint** | `GET https://plant-species-identifier.onrender.com/api/health` |
| **Prediction Endpoint** | `POST https://plant-species-identifier.onrender.com/api/predict` |
| **Inference Engine** | TFLite Interpreter (`tf.lite.Interpreter`) with Keras model fallback |
| **Cloud Deployment Status** | **`LIVE & VERIFIED` (HTTP 200 OK)** |

---

## 📌 1. Render Web Service Overview

Render is a unified cloud platform for hosting static sites, web applications, background workers, and REST microservices.

Our Python Flask backend (`backend/app.py`), MobileNetV2 Keras model (`model/leafscan_mobilenetv2.keras`), and TFLite model (`model/leafscan_mobilenetv2.tflite`) are deployed as a **Render Web Service**.

---

## ⚙️ 2. Production Service Configuration

| Setting Name | Configuration Value | Description |
|:---|:---|:---|
| **Service Name** | `plant-species-identifier` | Web Service instance name on Render |
| **Service Type** | Web Service | Python WSGI REST microservice |
| **Runtime Environment** | Python 3.11.8 | Configured via `.python-version` & `render.yaml` |
| **Build Command** | `python -m pip install --upgrade pip && pip install -r backend/requirements.txt` | Installs Flask, TensorFlow, Gunicorn, Pillow |
| **Start Command** | `gunicorn --chdir backend app:app --timeout 120 --workers 1 --threads 2 --bind 0.0.0.0:$PORT` | Production Gunicorn WSGI server |
| **Health Check Path** | `/api/health` | Automated Render health check monitoring |
| **Instance Type** | Free (512 MB RAM / 0.1 CPU) | Render Free Tier Web Service |

---

## 🔑 3. Environment Variables Configuration

The following environment variables are configured in the Render Dashboard (`Environment` tab):

| Variable Key | Value | Description |
|:---|:---|:---|
| `PYTHON_VERSION` | `3.11.8` | Forces Render to compile with Python 3.11 |
| `FRONTEND_ORIGIN` | `http://localhost:5173` | Production React PWA frontend origin for CORS |
| `PORT` | *(Injected by Render)* | Listens on Render assigned port (e.g. 10000) |

---

## 📂 4. Model File Path Resolution & Low-Memory TFLite Optimization

- **Model Weight Files**:
  - `model/leafscan_mobilenetv2.tflite` (**`8.53 MB`**) — Primary low-memory cloud inference engine.
  - `model/leafscan_mobilenetv2.keras` (**`9.34 MB`**) — Backup Keras model.
- **TFLite Cloud Strategy**: Loading standard Keras TensorFlow model into memory on Render's Free 512 MB RAM CPU instance consumed >400 MB RAM, causing HTTP 502 proxy timeouts. Using `tf.lite.Interpreter` inside `backend/app.py` reduced memory consumption to **< 20 MB** and inference latency to **~50ms**.
- **Git Tracking**: `.gitignore` tracks model weights directly (`!model/leafscan_mobilenetv2.tflite`, `!model/leafscan_mobilenetv2.keras`). Render pulls weights automatically upon git push.
- **Cross-Platform Path Resolution**: `backend/app.py` resolves paths dynamically using `BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))`.

---

## 🧪 5. Empirical Live Verification Test Results

Executed via [scratch/test_live_render_backend.py](file:///f:/DOC/My%20Projects/plant-species-identifier/scratch/test_live_render_backend.py):

| Test Step | Target Endpoint | HTTP Status | Response Data | Round-Trip Latency | Result |
|:---|:---|:---:|:---|:---:|:---:|
| **1. Health Check** | `GET /api/health` | **`200 OK`** | `model_loaded: true`, `model_type: "tflite"`, `num_classes: 10` | **689.93 ms** | **`PASS`** |
| **2. Real Prediction #1** | `POST /api/predict` (`Apple___healthy`) | **`200 OK`** | `species: "Apple___healthy"`, `confidence: 0.2885` (28.85%) | **1422.72 ms** | **`PASS`** |
| **3. Real Prediction #2** | `POST /api/predict` (`Potato___Late_blight`) | **`200 OK`** | `species: "Potato___Late_blight"`, `confidence: 0.3807` (38.07%) | **696.01 ms** | **`PASS`** |
| **4. Post-Prediction Health** | `GET /api/health` | **`200 OK`** | `model_loaded: true` (Persistent memory state) | **320.14 ms** | **`PASS`** |

---

## ⚡ 6. Render Cold-Start Behavior

On Render's Free Tier:
- **Inactivity Sleep**: The service spins down after 15 minutes of non-use.
- **Cold-Start Waking**: The first request after sleep takes ~15–20 seconds to spin up the container and load TFLite weights.
- **Warm Inference Speed**: Subsequent leaf prediction HTTP requests execute in **~600–700ms total round-trip time**.

