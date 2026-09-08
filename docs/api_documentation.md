# LeafScan — REST API Documentation

**Flask Backend Service Endpoints & Specification**

---

## 🌐 Base URL
```
Local Development: http://127.0.0.1:5000
Vite Proxy:        http://localhost:5173/api -> http://127.0.0.1:5000/api
```

---

## 📌 1. Health Check Endpoint

### `GET /api/health`

Returns operational status of the Flask backend API and verifies whether the Keras MobileNetV2 model is pre-loaded in memory.

#### Request Headers
```http
GET /api/health HTTP/1.1
Host: 127.0.0.1:5000
Accept: application/json
```

#### Response (HTTP 200 OK)
```json
{
  "status": "success",
  "message": "Plant Identification API is running",
  "model_loaded": true,
  "num_classes": 10
}
```

---

## 📌 2. Image Validation Endpoint

### `POST /api/upload`

Validates uploaded leaf image format (JPG, PNG, WEBP) and file size (< 5 MB).

#### Request Headers & Body
```http
POST /api/upload HTTP/1.1
Content-Type: multipart/form-data; boundary=----WebKitFormBoundary
```
- **Form Field**: `image` (or `file`): Image File Binary

#### Response (HTTP 200 OK)
```json
{
  "status": "success",
  "message": "Leaf image received and validated successfully by Flask API.",
  "filename": "sample_leaf.jpg",
  "content_type": "image/jpeg",
  "size_bytes": 145200,
  "size_formatted": "141.8 KB"
}
```

#### Error Response (HTTP 400 Bad Request)
```json
{
  "status": "error",
  "message": "Unsupported file format. Please upload a JPG, JPEG, PNG, or WEBP image."
}
```

---

## 📌 3. Real AI Prediction Endpoint

### `POST /api/predict`

Executes real MobileNetV2 deep learning inference on uploaded leaf photograph.

#### Request Body
- **Multipart Form Data**: `image` (or `file`)

#### Response (HTTP 200 OK)
```json
{
  "status": "success",
  "prediction": {
    "species": "Potato___Late_blight",
    "confidence": 0.3807
  },
  "top_predictions": [
    {
      "species": "Potato___Late_blight",
      "confidence": 0.3807
    },
    {
      "species": "Peach___Bacterial_spot",
      "confidence": 0.2571
    },
    {
      "species": "Grape___healthy",
      "confidence": 0.1547
    }
  ]
}
```

#### Error Status Codes
- **HTTP 400 Bad Request**: Missing file, invalid extension, or image exceeds 5 MB.
- **HTTP 500 Internal Server Error**: Keras model weights fail to load or unhandled processing error.
