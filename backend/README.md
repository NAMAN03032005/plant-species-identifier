# LeafScan — Backend Flask API Documentation

This directory contains the Python Flask microservice providing real-time AI image validation and MobileNetV2 Deep Learning model inference endpoints.

---

## 🚀 API Endpoints Summary

### 1. Health Check Endpoint

- **Route**: `GET /api/health`
- **Description**: Returns operational status of the Flask API server and confirms whether the MobileNetV2 model weights are loaded into memory.
- **Response Format**: `application/json`

**Example Response**:
```json
{
  "status": "success",
  "message": "Plant Identification API is running",
  "model_loaded": true,
  "num_classes": 10
}
```

---

### 2. Leaf Image Validation Endpoint

- **Route**: `POST /api/upload`
- **Description**: Validates image format (`.jpg`, `.jpeg`, `.png`, `.webp`) and file size limit (`<= 5 MB`).
- **Content-Type**: `multipart/form-data`
- **Form Field**: `image` or `file`

**Example Response**:
```json
{
  "status": "success",
  "message": "Leaf image received and validated successfully by Flask API.",
  "filename": "sample_leaf.jpg",
  "content_type": "image/jpeg",
  "size_bytes": 825,
  "size_formatted": "0.8 KB"
}
```

---

### 3. AI Species Prediction Endpoint

- **Route**: `POST /api/predict`
- **Description**: Performs real MobileNetV2 Transfer Learning inference on an uploaded leaf photograph. Loads image stream in memory, resizes to `224x224`, normalizes, and outputs predicted plant species class and top-3 confidence scores.
- **Method**: `POST`
- **Content-Type**: `multipart/form-data`
- **Form Field**: `image` (or `file`)

**Example Response**:
```json
{
  "status": "success",
  "prediction": {
    "species": "Apple___healthy",
    "confidence": 0.9412
  },
  "top_predictions": [
    {
      "species": "Apple___healthy",
      "confidence": 0.9412
    },
    {
      "species": "Apple___Black_rot",
      "confidence": 0.0385
    },
    {
      "species": "Cherry___healthy",
      "confidence": 0.0203
    }
  ]
}
```

**Error Responses**:
- `400 Bad Request` if no file attached, unsupported file format, or file size exceeds 5 MB limit.

```json
{
  "status": "error",
  "message": "Please upload a valid JPG, JPEG, PNG, or WEBP image."
}
```

---

## 🛠️ Model Loading Architecture

The Keras model weights ([leafscan_mobilenetv2.keras](file:///f:/DOC/My%20Projects/plant-species-identifier/model/leafscan_mobilenetv2.keras)) and class label mapping ([class_names.json](file:///f:/DOC/My%20Projects/plant-species-identifier/model/class_names.json)) are loaded **ONCE on application startup** into global memory variables rather than re-reading from disk per request. This optimizes inference latency for web requests.
