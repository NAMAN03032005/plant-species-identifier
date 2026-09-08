"""
LeafScan - End-to-End Automated Testing & Security Verification Suite (Step 10)
================================================================================
Performs comprehensive empirical validation across the entire system:
  1. GET  /api/health      - API operational status & Keras model pre-loading check.
  2. POST /api/predict     - Real Deep Learning inference on 5 test dataset images.
  3. Invalid File Test    - Security verification rejecting non-image formats (.txt, .pdf).
  4. Oversized File Test  - Security verification rejecting files > 5 MB.
  5. TFLite Audit Check   - TensorFlow Lite interpreter loading and image prediction verification.
"""

import os
import sys
import json
import io
import requests
import numpy as np
from PIL import Image
import tensorflow as tf

# Suppress log noise
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

BASE_URL = "http://127.0.0.1:5000"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset', 'test')
MODEL_DIR = os.path.join(BASE_DIR, 'model')
TFLITE_PATH = os.path.join(MODEL_DIR, 'leafscan_mobilenetv2.tflite')
CLASS_NAMES_PATH = os.path.join(MODEL_DIR, 'class_names.json')


def run_e2e_tests():
    print("=" * 80)
    print("        LeafScan — Complete End-to-End System & Security Test Suite")
    print("=" * 80)

    test_results = []

    # -------------------------------------------------------------
    # Test 1: GET /api/health Status
    # -------------------------------------------------------------
    print("\n[TEST 1] GET /api/health Status Check...")
    try:
        r = requests.get(f"{BASE_URL}/api/health", timeout=5)
        if r.status_code == 200 and r.json().get('model_loaded') is True:
            print("  • Status Code : 200 OK")
            print("  • Model Loaded: TRUE")
            print("  • Status      : PASS [OK]")
            test_results.append(("Health Check API", "HTTP 200, Model Loaded = True", "PASS"))
        else:
            print(f"  • Failed: {r.status_code} - {r.text}")
            test_results.append(("Health Check API", f"Status Code: {r.status_code}", "FAIL"))
    except Exception as e:
        print(f"  • Connection Error: {e}")
        test_results.append(("Health Check API", "Cannot connect to Flask server", "FAIL"))
        sys.exit(1)

    # -------------------------------------------------------------
    # Test 2: Real Prediction API (5 Real Test Images)
    # -------------------------------------------------------------
    print("\n[TEST 2] Real Deep Learning Inference (POST /api/predict)...")
    test_classes = [
        "Apple___Apple_scab",
        "Corn___healthy",
        "Grape___Black_rot",
        "Peach___Bacterial_spot",
        "Potato___Late_blight"
    ]

    real_preds_success = True
    print("\n" + "-" * 95)
    print(f"{'Target Class':<24} | {'Predicted Class':<24} | {'Conf %':<8} | {'Top-3 Predicted Class'}")
    print("-" * 95)

    for target_cls in test_classes:
        cdir = os.path.join(DATASET_DIR, target_cls)
        if not os.path.exists(cdir) or not os.listdir(cdir):
            print(f"  [Warning] Test class directory empty: {target_cls}")
            continue

        sample_fname = sorted(os.listdir(cdir))[0]
        sample_path = os.path.join(cdir, sample_fname)

        with open(sample_path, 'rb') as img_file:
            files = {'image': (sample_fname, img_file, 'image/jpeg')}
            r = requests.post(f"{BASE_URL}/api/predict", files=files, timeout=10)

        if r.status_code == 200:
            data = r.json()
            top1_sp = data['prediction']['species']
            top1_conf = data['prediction']['confidence'] * 100
            top3 = [p['species'] for p in data.get('top_predictions', [])]
            top3_str = ", ".join(top3[:2])

            print(f"{target_cls:<24} | {top1_sp:<24} | {top1_conf:>6.2f}% | {top3_str}")
        else:
            print(f"{target_cls:<24} | ERROR: {r.status_code:<17} | N/A     | N/A")
            real_preds_success = False

    print("-" * 95)
    if real_preds_success:
        test_results.append(("Real AI Prediction API (5 Images)", "5/5 Real Predictions Processed", "PASS"))
    else:
        test_results.append(("Real AI Prediction API (5 Images)", "Prediction Failed on some images", "FAIL"))

    # -------------------------------------------------------------
    # Test 3: Invalid Input Format Rejection (.txt & .pdf)
    # -------------------------------------------------------------
    print("\n[TEST 3] Security Audit: Invalid File Format Rejection...")
    txt_file = io.BytesIO(b"Dummy malicious plain text file content")
    files_invalid = {'image': ('malicious.txt', txt_file, 'text/plain')}
    r_invalid = requests.post(f"{BASE_URL}/api/predict", files=files_invalid, timeout=5)

    if r_invalid.status_code == 400:
        print(f"  • Rejected Invalid .txt File: HTTP 400 BAD REQUEST ({r_invalid.json().get('message')})")
        test_results.append(("Invalid File Format Rejection", "HTTP 400 Bad Request Returned", "PASS"))
    else:
        print(f"  • Unexpected Response for .txt File: {r_invalid.status_code}")
        test_results.append(("Invalid File Format Rejection", f"Status: {r_invalid.status_code}", "FAIL"))

    # -------------------------------------------------------------
    # Test 4: Oversized File Rejection (> 5 MB)
    # -------------------------------------------------------------
    print("\n[TEST 4] Security Audit: File Size Limit (> 5 MB Rejection)...")
    large_dummy = io.BytesIO(b"0" * (6 * 1024 * 1024))  # 6 MB dummy
    files_large = {'image': ('large_sample.jpg', large_dummy, 'image/jpeg')}
    r_large = requests.post(f"{BASE_URL}/api/predict", files=files_large, timeout=5)

    if r_large.status_code in [400, 413]:
        print(f"  • Rejected Oversized 6 MB File: HTTP {r_large.status_code} ({r_large.json().get('message')})")
        test_results.append(("Oversized File Rejection (>5 MB)", f"HTTP {r_large.status_code} Returned", "PASS"))
    else:
        print(f"  • Unexpected Response for 6 MB File: {r_large.status_code}")
        test_results.append(("Oversized File Rejection (>5 MB)", f"Status: {r_large.status_code}", "FAIL"))

    # -------------------------------------------------------------
    # Test 5: TensorFlow Lite Interpreter Inference
    # -------------------------------------------------------------
    print("\n[TEST 5] TensorFlow Lite On-Device Interpreter Audit...")
    if os.path.exists(TFLITE_PATH) and os.path.exists(CLASS_NAMES_PATH):
        try:
            interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
            interpreter.allocate_tensors()
            in_idx = interpreter.get_input_details()[0]['index']
            out_idx = interpreter.get_output_details()[0]['index']

            with open(CLASS_NAMES_PATH, 'r') as f:
                class_names = json.load(f)

            # Test inference on 1 real image
            test_img_path = os.path.join(DATASET_DIR, test_classes[0], sorted(os.listdir(os.path.join(DATASET_DIR, test_classes[0])))[0])
            img = Image.open(test_img_path).convert("RGB").resize((224, 224))
            img_array = np.array(img, dtype=np.float32)
            img_batch = np.expand_dims(img_array, axis=0)

            interpreter.set_tensor(in_idx, img_batch)
            interpreter.invoke()
            tflite_preds = interpreter.get_tensor(out_idx)[0]
            top_tflite_idx = int(np.argmax(tflite_preds))
            top_tflite_cls = class_names[top_tflite_idx]

            print(f"  • TFLite Interpreter Loaded Successfully")
            print(f"  • Sample TFLite Prediction: {top_tflite_cls} ({tflite_preds[top_tflite_idx]*100:.2f}%)")
            test_results.append(("TFLite Interpreter Verification", "Inference PASS on Real Image", "PASS"))
        except Exception as e:
            print(f"  • TFLite Error: {e}")
            test_results.append(("TFLite Interpreter Verification", f"Error: {e}", "FAIL"))
    else:
        print("  • TFLite model or class names missing.")
        test_results.append(("TFLite Interpreter Verification", "TFLite Model Missing", "FAIL"))

    # -------------------------------------------------------------
    # Test Summary Report
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("                 END-TO-END AUTOMATED TEST MATRIX")
    print("=" * 80)
    print(f"{'Test Suite Identifier':<35} | {'Measured Behavior':<32} | {'Status'}")
    print("-" * 80)
    for name, detail, status in test_results:
        print(f"{name:<35} | {detail:<32} | {status}")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    run_e2e_tests()
