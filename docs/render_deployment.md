# LeafScan — Render Web Service Deployment Guide (Step 11A)

**Production Cloud Hosting Guide for Python Flask API Microservice**

---

## 📌 1. Render Web Service Overview

Render is a unified cloud platform for hosting static sites, web applications, background workers, and REST microservices.

Our Python Flask backend (`backend/app.py`) and MobileNetV2 Keras model (`model/leafscan_mobilenetv2.keras`) are configured to deploy as a **Render Web Service**.

---

## ⚙️ 2. Production Service Configuration

| Setting Name | Recommended Configuration Value | Description |
|:---|:---|:---|
| **Service Type** | Web Service | Python WSGI application |
| **Runtime Environment** | Python 3.11 | Python version compatibility |
| **Build Command** | `pip install -r backend/requirements.txt` | Installs Flask, TensorFlow, Gunicorn, Pillow |
| **Start Command** | `gunicorn --chdir backend app:app --bind 0.0.0.0:$PORT` | Production Gunicorn WSGI server |
| **Health Check Path** | `/api/health` | Automated Render health check monitoring |
| **Instance Type** | Free / Starter | Minimum 512 MB RAM instance |

---

## 🔑 3. Environment Variables Configuration

Configure the following environment variables in your Render Dashboard (`Environment` tab):

| Variable Key | Example Value | Description |
|:---|:---|:---|
| `PYTHON_VERSION` | `3.11.8` | Forces Render to compile with Python 3.11 |
| `FRONTEND_ORIGIN` | `https://leafscan-frontend.onrender.com` | Production React PWA frontend origin for CORS |
| `PORT` | *(Automatically injected by Render)* | Listens on Render assigned port (e.g. 10000) |

---

## 📂 4. Model File Path Resolution & Git Tracking

- **Model Weight File**: `model/leafscan_mobilenetv2.keras` (**`9.34 MB`**).
- **Git Tracking**: `.gitignore` has been updated (`!model/leafscan_mobilenetv2.keras`) so model weights are tracked directly in Git. Render pulls model weights automatically upon git push without requiring external S3 downloads.
- **Cross-Platform Path Resolution**: `backend/app.py` resolves paths dynamically using `BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))`, avoiding hardcoded Windows backslashes (`\`).

---

## ⚡ 5. Expected Render Cold-Start Behavior

On Render's Free Tier:
- **Inactivity Sleep**: The service spins down after 15 minutes of non-use.
- **Cold-Start Delay**: The first request (`GET /api/health` or `POST /api/predict`) after sleep may take ~15–30 seconds to wake up the service and load TensorFlow weights into memory.
- **Warm Inference Speed**: Subsequent leaf predictions execute in **~30–50ms**.

---

## 🛠️ 6. Deployment Workflow Options

### Option A: Render Blueprint (Recommended)
Render automatically detects `render.yaml` in your root repository. Simply connect your GitHub repository and click **Apply Blueprint**.

### Option B: Manual Web Service Setup
1. Log into [Render Dashboard](https://dashboard.render.com/).
2. Click **New +** -> **Web Service**.
3. Connect your `plant-species-identifier` repository.
4. Set **Build Command**: `pip install -r backend/requirements.txt`
5. Set **Start Command**: `gunicorn --chdir backend app:app --bind 0.0.0.0:$PORT`
6. Add Environment Variable `FRONTEND_ORIGIN` with your frontend URL.
7. Click **Create Web Service**.
