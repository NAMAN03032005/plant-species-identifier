"""
LeafScan - Backend Prediction API Integration Test (Step 5)
===========================================================
This test script verifies the POST /api/predict endpoint using the Flask test client:
  1. Loads a real test leaf image from dataset/test/.
  2. Sends a POST request to /api/predict with key 'image'.
  3. Validates HTTP status 200 OK and response JSON schema.
"""

import os
import sys
import io
import json

# Ensure app.py can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app import app

# Real test image path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_IMG_PATH = os.path.join(BASE_DIR, 'dataset', 'test', 'Apple___healthy', 'Apple___healthy_sample_035.jpg')


def test_predict_endpoint():
    print("=" * 65)
    print("      Testing Flask REST API - POST /api/predict Endpoint")
    print("=" * 65)

    if not os.path.exists(TEST_IMG_PATH):
        print(f"[Error] Test image not found at: {TEST_IMG_PATH}")
        sys.exit(1)

    client = app.test_client()

    # 1. Test Health Endpoint
    health_res = client.get('/api/health')
    print("Health Check Status Code:", health_res.status_code)
    print("Health Check Data      :", health_res.get_json())
    assert health_res.status_code == 200
    assert health_res.get_json().get("model_loaded") is True

    # 2. Test Real Image Prediction Endpoint
    with open(TEST_IMG_PATH, 'rb') as f:
        img_bytes = f.read()

    data = {
        'image': (io.BytesIO(img_bytes), 'Apple___healthy_sample_035.jpg')
    }

    predict_res = client.post(
        '/api/predict',
        data=data,
        content_type='multipart/form-data'
    )

    print("\nPredict Endpoint Status Code:", predict_res.status_code)
    json_data = predict_res.get_json()
    print("Predict Endpoint JSON Output :\n", json.dumps(json_data, indent=2))

    # Assertions
    assert predict_res.status_code == 200
    assert json_data.get("status") == "success"
    assert "prediction" in json_data
    assert "species" in json_data["prediction"]
    assert "confidence" in json_data["prediction"]
    assert "top_predictions" in json_data
    assert len(json_data["top_predictions"]) == 3

    print("\n[Success] Flask POST /api/predict endpoint test passed with 100% compliance!")
    print("=" * 65)


if __name__ == '__main__':
    test_predict_endpoint()
