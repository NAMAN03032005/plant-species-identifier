"""
LeafScan - MobileNetV2 to TensorFlow Lite Conversion Script (Step 8)
======================================================================
This script converts the trained Keras MobileNetV2 model into TensorFlow Lite format (.tflite)
for optimized mobile and edge deployment.

Conversions:
  1. Standard TensorFlow Lite model (Float32): model/leafscan_mobilenetv2.tflite
  2. Float16 Quantized TensorFlow Lite model: model/leafscan_mobilenetv2_float16.tflite

Features:
  - Preserves original model architecture and weights.
  - Reports original vs TFLite file sizes and percentage reduction.
"""

import os
import sys
import tensorflow as tf

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'model')
KERAS_MODEL_PATH = os.path.join(MODEL_DIR, 'leafscan_mobilenetv2.keras')
TFLITE_PATH = os.path.join(MODEL_DIR, 'leafscan_mobilenetv2.tflite')
TFLITE_FP16_PATH = os.path.join(MODEL_DIR, 'leafscan_mobilenetv2_float16.tflite')


def convert_model():
    print("=" * 70)
    print("      LeafScan MobileNetV2 -> TensorFlow Lite Conversion")
    print("=" * 70)

    if not os.path.exists(KERAS_MODEL_PATH):
        print(f"[Error] Keras model file not found at: {KERAS_MODEL_PATH}")
        sys.exit(1)

    # 1. Load Keras Model
    print(f"\n[1/4] Loading Keras model from: {KERAS_MODEL_PATH}")
    model = tf.keras.models.load_model(KERAS_MODEL_PATH)
    keras_size_bytes = os.path.getsize(KERAS_MODEL_PATH)
    keras_size_mb = keras_size_bytes / (1024 * 1024)
    print(f"  • Keras Model Size       : {keras_size_mb:.2f} MB ({keras_size_bytes:,} bytes)")
    print(f"  • Model Parameters       : {model.count_params():,}")
    print(f"  • Model Input Shape      : {model.input_shape}")
    print(f"  • Model Output Shape     : {model.output_shape}")

    # 2. Convert to Standard TensorFlow Lite (Float32)
    print(f"\n[2/4] Converting to Standard TensorFlow Lite (.tflite)...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    with open(TFLITE_PATH, 'wb') as f:
        f.write(tflite_model)

    tflite_size_bytes = os.path.getsize(TFLITE_PATH)
    tflite_size_mb = tflite_size_bytes / (1024 * 1024)
    tflite_reduction = ((keras_size_bytes - tflite_size_bytes) / keras_size_bytes) * 100

    print(f"  • Standard TFLite Size   : {tflite_size_mb:.2f} MB ({tflite_size_bytes:,} bytes)")
    print(f"  • Size Reduction         : {tflite_reduction:.2f}%")

    # 3. Convert to Float16 Quantized TensorFlow Lite
    print(f"\n[3/4] Converting to Float16 Quantized TensorFlow Lite (_float16.tflite)...")
    converter_fp16 = tf.lite.TFLiteConverter.from_keras_model(model)
    converter_fp16.optimizations = [tf.lite.Optimize.DEFAULT]
    converter_fp16.target_spec.supported_types = [tf.float16]
    
    fp16_path = TFLITE_FP16_PATH
    try:
        tflite_fp16_model = converter_fp16.convert()
        with open(fp16_path, 'wb') as f:
            f.write(tflite_fp16_model)

        fp16_size_bytes = os.path.getsize(fp16_path)
        fp16_size_mb = fp16_size_bytes / (1024 * 1024)
        fp16_reduction = ((keras_size_bytes - fp16_size_bytes) / keras_size_bytes) * 100

        print(f"  • Float16 TFLite Size    : {fp16_size_mb:.2f} MB ({fp16_size_bytes:,} bytes)")
        print(f"  • Float16 Size Reduction : {fp16_reduction:.2f}% vs Keras model")
    except Exception as e:
        print(f"  [Warning] Float16 conversion note: {e}")
        print("  Using standard TFLite file as primary mobile model.")
        fp16_path = TFLITE_PATH
        fp16_size_mb = tflite_size_mb
        fp16_reduction = tflite_reduction

    # 4. Summary Table
    print("\n" + "=" * 70)
    print("                    Conversion Summary")
    print("=" * 70)
    print(f"{'Format':<30} | {'Size (MB)':<12} | {'Size Reduction':<15}")
    print("-" * 70)
    print(f"{'Original Keras (.keras)':<30} | {keras_size_mb:>10.2f} MB | {'0.00%':>15}")
    print(f"{'Standard TFLite (.tflite)':<30} | {tflite_size_mb:>10.2f} MB | {f'{tflite_reduction:.2f}%':>15}")
    if os.path.exists(fp16_path) and fp16_path != TFLITE_PATH:
        print(f"{'Float16 TFLite (_float16.tflite)':<30} | {fp16_size_mb:>10.2f} MB | {f'{fp16_reduction:.2f}%':>15}")
    print("=" * 70)
    print("TFLite conversion completed successfully.\n")


if __name__ == '__main__':
    convert_model()
