"""
LeafScan - Mobile TFLite Standalone Inference Demonstration (Step 8)
======================================================================
This CLI tool demonstrates on-device TensorFlow Lite inference on leaf images
without requiring a web server or Flask REST API.

Usage:
  python scripts/mobile_inference_demo.py <image_path>

Example:
  python scripts/mobile_inference_demo.py dataset/test/Apple___healthy/Apple___healthy_sample_001.jpg
"""

import os
import sys
import json
import numpy as np
from PIL import Image
import tensorflow as tf

# Suppress log noise
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'model')
TFLITE_FP16_PATH = os.path.join(MODEL_DIR, 'leafscan_mobilenetv2_float16.tflite')
TFLITE_STD_PATH = os.path.join(MODEL_DIR, 'leafscan_mobilenetv2.tflite')
CLASS_NAMES_PATH = os.path.join(MODEL_DIR, 'class_names.json')


def parse_label(raw_label):
    """Split raw class label (e.g. 'Apple___Black_rot') into species and leaf condition."""
    if "___" in raw_label:
        parts = raw_label.split("___", 1)
        species = parts[0].replace("_", " ").strip()
        condition = parts[1].replace("_", " ").title().strip()
    else:
        species = raw_label.replace("_", " ").strip()
        condition = "Unknown"
    return species, condition


def run_mobile_demo(image_path):
    print("=" * 65)
    print("        LeafScan — On-Device Mobile TFLite Demonstration")
    print("=" * 65)

    if not os.path.exists(image_path):
        print(f"[Error] Image file not found: {image_path}")
        sys.exit(1)

    # 1. Determine TFLite model file
    if os.path.exists(TFLITE_STD_PATH):
        tflite_model_path = TFLITE_STD_PATH
        model_type = "Standard Float32 TFLite (100% Parity)"
    elif os.path.exists(TFLITE_FP16_PATH):
        tflite_model_path = TFLITE_FP16_PATH
        model_type = "Float16 Quantized TFLite"
    else:
        print("[Error] No TFLite model file found. Run scripts/convert_to_tflite.py first.")
        sys.exit(1)

    if not os.path.exists(CLASS_NAMES_PATH):
        print(f"[Error] Class names file not found at: {CLASS_NAMES_PATH}")
        sys.exit(1)

    with open(CLASS_NAMES_PATH, 'r') as f:
        class_names = json.load(f)

    # 2. Load TFLite Model into Interpreter
    print(f"\n[1/3] Loading TFLite Engine ({model_type})...")
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # 3. Preprocess Input Image
    print(f"[2/3] Loading and Preprocessing Image: {os.path.basename(image_path)}")
    try:
        img = Image.open(image_path).convert("RGB").resize((224, 224))
        img_array = np.array(img, dtype=np.float32)
        img_batch = np.expand_dims(img_array, axis=0)
    except Exception as e:
        print(f"[Error] Failed to read/preprocess image: {e}")
        sys.exit(1)

    # 4. Perform On-Device Mobile TFLite Inference
    print("[3/3] Running TFLite Inference...")
    interpreter.set_tensor(input_details[0]['index'], img_batch)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]['index'])[0]

    # Rank top-3 predictions
    top_indices = np.argsort(preds)[::-1][:3]

    top1_idx = top_indices[0]
    top1_label = class_names[top1_idx]
    top1_conf = float(preds[top1_idx]) * 100
    top1_species, top1_condition = parse_label(top1_label)

    # Display Output
    print("\n" + "=" * 65)
    print("                    PREDICTION RESULTS")
    print("=" * 65)
    print(f"  • Top-1 Species Class    : {top1_species}")
    print(f"  • Leaf Condition         : {top1_condition}")
    print(f"  • Confidence Score       : {top1_conf:.2f}%")
    print(f"  • Raw Model Label        : {top1_label}")
    print("-" * 65)
    print("  • Top 3 Candidates       :")
    for rank, idx in enumerate(top_indices, 1):
        lbl = class_names[idx]
        sp, cond = parse_label(lbl)
        conf = float(preds[idx]) * 100
        print(f"     {rank}. [{conf:>6.2f}%] {sp} — {cond} ({lbl})")
    print("=" * 65 + "\n")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python scripts/mobile_inference_demo.py <image_path>")
        print("Example: python scripts/mobile_inference_demo.py dataset/test/Apple___healthy/Apple___healthy_sample_001.jpg")
        sys.exit(1)

    run_mobile_demo(sys.argv[1])
