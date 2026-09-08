"""
LeafScan - Local Backend Production Test & Startup Benchmark (Step 11A)
========================================================================
Measures:
  1. Model loading time at startup.
  2. First GET /api/health response time.
  3. First POST /api/predict response time on 2 real test images.
"""

import os
import sys
import time
import json
import requests

BASE_URL = "http://127.0.0.1:5000"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset', 'test')


def run_benchmark():
    print("=" * 70)
    print("      LeafScan — Backend Production Startup & Inference Benchmark")
    print("=" * 70)

    # 1. Measure Health Response
    t0 = time.time()
    r = requests.get(f"{BASE_URL}/api/health", timeout=10)
    t1 = time.time()
    health_ms = (t1 - t0) * 1000

    print(f"\n[1/3] GET /api/health Check:")
    print(f"  • Status Code         : {r.status_code}")
    print(f"  • Health Response Time : {health_ms:.2f} ms")
    print(f"  • Response JSON       : {r.json()}")

    # 2. Test Real Prediction 1 (Apple Healthy)
    cdir1 = os.path.join(DATASET_DIR, 'Apple___healthy')
    test_img1 = os.path.join(cdir1, sorted(os.listdir(cdir1))[0])
    print(f"\n[2/3] Real Prediction 1 ({os.path.basename(test_img1)}):")
    t0 = time.time()
    with open(test_img1, 'rb') as f:
        r1 = requests.post(f"{BASE_URL}/api/predict", files={'image': ('sample1.jpg', f, 'image/jpeg')})
    t1 = time.time()
    pred1_ms = (t1 - t0) * 1000
    p1 = r1.json().get('prediction', {})
    print(f"  • Status Code         : {r1.status_code}")
    print(f"  • Prediction 1 Time   : {pred1_ms:.2f} ms")
    print(f"  • Predicted Class     : {p1.get('species')}")
    print(f"  • Confidence Score    : {p1.get('confidence')*100:.2f}%")

    # 3. Test Real Prediction 2 (Potato Late Blight)
    cdir2 = os.path.join(DATASET_DIR, 'Potato___Late_blight')
    test_img2 = os.path.join(cdir2, sorted(os.listdir(cdir2))[0])
    print(f"\n[3/3] Real Prediction 2 ({os.path.basename(test_img2)}):")
    t0 = time.time()
    with open(test_img2, 'rb') as f:
        r2 = requests.post(f"{BASE_URL}/api/predict", files={'image': ('sample2.jpg', f, 'image/jpeg')})
    t1 = time.time()
    pred2_ms = (t1 - t0) * 1000
    p2 = r2.json().get('prediction', {})
    print(f"  • Status Code         : {r2.status_code}")
    print(f"  • Prediction 2 Time   : {pred2_ms:.2f} ms")
    print(f"  • Predicted Class     : {p2.get('species')}")
    print(f"  • Confidence Score    : {p2.get('confidence')*100:.2f}%")

    print("\n" + "=" * 70)
    print("                    Benchmark Results Summary")
    print("=" * 70)
    print(f"  • Health Endpoint Latency    : {health_ms:.2f} ms")
    print(f"  • First Real Prediction Time : {pred1_ms:.2f} ms")
    print(f"  • Second Real Prediction Time: {pred2_ms:.2f} ms")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    run_benchmark()
