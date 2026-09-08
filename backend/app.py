"""
LeafScan - Backend Flask API (Step 5 - Real AI Inference Service)
=================================================================
This Flask server provides real-time Deep Learning leaf species identification:
  1. GET  /api/health  - Health check endpoint returning API & model status.
  2. POST /api/upload  - Image file validation & metadata verification endpoint.
  3. POST /api/predict - Real MobileNetV2 AI prediction endpoint.

Features:
  - Model (model/leafscan_mobilenetv2.keras) and labels (model/class_names.json)
    are loaded ONCE at server startup for fast inference.
  - Image preprocessing matches model training (224x224 RGB via Pillow).
  - Returns top-1 prediction species name and top-3 confidence probabilities.
"""

import os
import io
import sys
import json
import numpy as np
from PIL import Image
from flask import Flask, jsonify, request
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Suppress unnecessary TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

# Initialize Flask Application
app = Flask(__name__)

# Configure CORS to allow local React development frontend origin and production Render static site origins
allowed_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:5000",
    "https://plant-species-identifier-1.onrender.com",
    "https://plant-species-identifier-frontend.onrender.com"
]
frontend_origin = os.environ.get('FRONTEND_ORIGIN')
if frontend_origin:
    allowed_origins.append(frontend_origin.rstrip('/'))

CORS(app, resources={r"/api/*": {"origins": allowed_origins}})

# Robust Project Relative Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH_KERAS = os.path.join(BASE_DIR, 'model', 'leafscan_mobilenetv2.keras')
MODEL_PATH_TFLITE = os.path.join(BASE_DIR, 'model', 'leafscan_mobilenetv2.tflite')
CLASS_NAMES_PATH = os.path.join(BASE_DIR, 'model', 'class_names.json')

# File Validation Constraints
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'webp'}
MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024  # 5 MB limit

# Module Globals for Loaded Model and Labels
MODEL_KERAS = None
TFLITE_INTERPRETER = None
TFLITE_INPUT_DETAILS = None
TFLITE_OUTPUT_DETAILS = None
MODEL_TYPE = None  # 'tflite' or 'keras'
CLASS_NAMES = []


def load_model_and_labels():
    """
    Load species class labels and ML model ONCE at application startup.
    Tries TFLite interpreter first (ultra-fast, low memory footprint for cloud deployment),
    with automatic fallback to full TensorFlow Keras model.
    """
    global MODEL_KERAS, TFLITE_INTERPRETER, TFLITE_INPUT_DETAILS, TFLITE_OUTPUT_DETAILS, MODEL_TYPE, CLASS_NAMES

    if not os.path.exists(CLASS_NAMES_PATH):
        print(f"[Error] Class names JSON not found at: {CLASS_NAMES_PATH}")
        return False

    try:
        with open(CLASS_NAMES_PATH, 'r') as f:
            CLASS_NAMES = json.load(f)
        print(f"[Startup] Successfully loaded {len(CLASS_NAMES)} species classes.")
    except Exception as e:
        print(f"[Error] Failed to load class names JSON: {e}")
        return False

    # 1. Try loading TFLite model first (ultra-fast & < 20 MB RAM usage)
    if os.path.exists(MODEL_PATH_TFLITE):
        try:
            print(f"[Startup] Loading TFLite model from {MODEL_PATH_TFLITE}...")
            TFLITE_INTERPRETER = tf.lite.Interpreter(model_path=MODEL_PATH_TFLITE)
            TFLITE_INTERPRETER.allocate_tensors()
            TFLITE_INPUT_DETAILS = TFLITE_INTERPRETER.get_input_details()
            TFLITE_OUTPUT_DETAILS = TFLITE_INTERPRETER.get_output_details()
            MODEL_TYPE = 'tflite'
            
            # Execute warm-up prediction at startup
            print("[Startup] Executing TFLite warm-up prediction...")
            dummy_input = np.zeros((1, 224, 224, 3), dtype=np.float32)
            TFLITE_INTERPRETER.set_tensor(TFLITE_INPUT_DETAILS[0]['index'], dummy_input)
            TFLITE_INTERPRETER.invoke()
            _ = TFLITE_INTERPRETER.get_tensor(TFLITE_OUTPUT_DETAILS[0]['index'])
            print("[Startup] TFLite model initialized and warmed up successfully.")
            return True
        except Exception as e:
            print(f"[Startup Warning] Could not load TFLite model: {e}")

    # 2. Fallback to Keras model
    if os.path.exists(MODEL_PATH_KERAS):
        try:
            print(f"[Startup] Loading MobileNetV2 Keras model from {MODEL_PATH_KERAS}...")
            MODEL_KERAS = tf.keras.models.load_model(MODEL_PATH_KERAS)
            MODEL_TYPE = 'keras'
            
            print("[Startup] Executing Keras model warm-up prediction...")
            dummy_input = np.zeros((1, 224, 224, 3), dtype=np.float32)
            _ = MODEL_KERAS(dummy_input, training=False)
            print("[Startup] Keras model warm-up completed successfully.")
            return True
        except Exception as e:
            print(f"[Error] Failed to load Keras model: {e}")
            return False

    print("[Error] No model file (.tflite or .keras) found in model directory.")
    return False


# Execute model loading at server initialization
MODEL_LOADED = load_model_and_labels()


def allowed_file(filename):
    """Check if file has an allowed image extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/api/health', methods=['GET'])
def health_check():
    """
    Health Check Endpoint
    ----------------------
    Returns operational status of API service and model loading state.
    """
    return jsonify({
        "status": "success",
        "message": "Plant Identification API is running",
        "model_loaded": MODEL_TYPE is not None,
        "model_type": MODEL_TYPE,
        "num_classes": len(CLASS_NAMES) if CLASS_NAMES else 0
    }), 200


@app.route('/api/upload', methods=['POST'])
def upload_leaf_image():
    """
    Leaf Image Upload Validation Endpoint
    --------------------------------------
    Validates uploaded image format and file size (< 5 MB).
    """
    file = request.files.get('file') or request.files.get('image')

    if not file or file.filename == '':
        return jsonify({
            "status": "error",
            "message": "No file selected for upload."
        }), 400

    if not allowed_file(file.filename):
        return jsonify({
            "status": "error",
            "message": "Unsupported file format. Please upload a JPG, JPEG, PNG, or WEBP image."
        }), 400

    file.seek(0, os.SEEK_END)
    file_length = file.tell()
    file.seek(0)

    if file_length > MAX_FILE_SIZE_BYTES:
        return jsonify({
            "status": "error",
            "message": f"Image size ({round(file_length / (1024 * 1024), 2)} MB) exceeds the 5 MB limit."
        }), 400

    filename = secure_filename(file.filename)
    content_type = file.content_type or "image/unknown"
    size_formatted = f"{round(file_length / 1024, 1)} KB" if file_length < 1024 * 1024 else f"{round(file_length / (1024 * 1024), 2)} MB"

    return jsonify({
        "status": "success",
        "message": "Leaf image received and validated successfully by Flask API.",
        "filename": filename,
        "content_type": content_type,
        "size_bytes": file_length,
        "size_formatted": size_formatted
    }), 200


@app.route('/api/predict', methods=['POST'])
def predict_plant_species():
    """
    Real AI Leaf Species Prediction Endpoint
    ----------------------------------------
    Accepts multipart/form-data image under key 'image' (or 'file').
    Preprocesses image to 224x224 RGB, performs MobileNetV2 model inference,
    and returns top-1 predicted species name and top-3 confidence scores.
    """
    if MODEL_TYPE is None or not CLASS_NAMES:
        return jsonify({
            "status": "error",
            "message": "Model is not loaded on server. Please check model files."
        }), 500

    # 1. Extract image file part
    file = request.files.get('image') or request.files.get('file')

    if not file or file.filename == '':
        return jsonify({
            "status": "error",
            "message": "Please select an image file to analyze."
        }), 400

    # 2. Check file format extension
    if not allowed_file(file.filename):
        return jsonify({
            "status": "error",
            "message": "Please upload a valid JPG, JPEG, PNG, or WEBP image."
        }), 400

    # 3. Read image stream into memory safely
    try:
        image_bytes = file.read()
        
        if len(image_bytes) > MAX_FILE_SIZE_BYTES:
            return jsonify({
                "status": "error",
                "message": "Image size must be less than 5 MB."
            }), 400

        # Process image in-memory using Pillow
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_resized = img.resize((224, 224))
        img_array = np.array(img_resized, dtype=np.float32)
        img_batch = np.expand_dims(img_array, axis=0)

        # 4. Perform Real Deep Learning Inference
        if MODEL_TYPE == 'tflite':
            TFLITE_INTERPRETER.set_tensor(TFLITE_INPUT_DETAILS[0]['index'], img_batch)
            TFLITE_INTERPRETER.invoke()
            preds = TFLITE_INTERPRETER.get_tensor(TFLITE_OUTPUT_DETAILS[0]['index'])[0]
        else:
            preds = MODEL_KERAS(img_batch, training=False).numpy()[0]

        top1_idx = int(np.argmax(preds))
        top1_species = CLASS_NAMES[top1_idx]
        top1_confidence = round(float(preds[top1_idx]), 4)

        # Calculate Top-3 Probabilities
        top3_indices = np.argsort(preds)[::-1][:3]
        top_predictions = []
        for idx in top3_indices:
            top_predictions.append({
                "species": CLASS_NAMES[idx],
                "confidence": round(float(preds[idx]), 4)
            })

        # 5. Return JSON Response
        return jsonify({
            "status": "success",
            "prediction": {
                "species": top1_species,
                "confidence": top1_confidence
            },
            "top_predictions": top_predictions,
            "engine": MODEL_TYPE
        }), 200

    except Exception as e:
        print(f"[Prediction Error] Exception during inference: {e}")
        return jsonify({
            "status": "error",
            "message": "Failed to process image for prediction. Please try another image."
        }), 400


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting LeafScan Flask API server on http://0.0.0.0:{port}...")
    app.run(host='0.0.0.0', port=port, debug=False)

