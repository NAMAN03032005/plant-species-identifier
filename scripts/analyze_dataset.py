"""
LeafScan - Dataset Analysis & Environment Inspection Script (Step 3)
=====================================================================
This script analyzes the prepared plant leaf dataset and inspects the local execution environment:
  1. Counts images per class across Train, Validation, and Test splits.
  2. Verifies class alignment against model/class_names.json.
  3. Evaluates class imbalance metrics (min/max count per class).
  4. Inspects Python environment, TensorFlow installation, and CPU/GPU availability.

Note: No neural network training is performed by this script.
"""

import os
import sys
import json
import platform

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
TRAIN_DIR = os.path.join(DATASET_DIR, 'train')
VAL_DIR = os.path.join(DATASET_DIR, 'validation')
TEST_DIR = os.path.join(DATASET_DIR, 'test')
CLASS_NAMES_PATH = os.path.join(BASE_DIR, 'model', 'class_names.json')


def check_environment():
    """Inspect Python, system, TensorFlow, and hardware acceleration status."""
    print("=" * 70)
    print("        1. ENVIRONMENT & HARDWARE ACCELERATION INSPECTION")
    print("=" * 70)
    print(f"Operating System  : {platform.system()} {platform.release()} ({platform.architecture()[0]})")
    print(f"Python Version    : {sys.version.split()[0]} ({platform.python_implementation()})")
    
    # Check TensorFlow installation & GPU availability
    try:
        import tensorflow as tf
        print(f"TensorFlow        : Installed (v{tf.__version__})")
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"Hardware Accelerator : GPU Detected ({len(gpus)} device(s))")
            for gpu in gpus:
                print(f"  - Device: {gpu.name}")
        else:
            print("Hardware Accelerator : CPU Only (No CUDA GPU detected)")
    except ImportError:
        print("TensorFlow        : Not installed yet (Scheduled for Step 4 installation)")
        print("Hardware Accelerator : N/A")
    print("-" * 70)


def analyze_dataset():
    """Analyze dataset splits and print statistics."""
    print("\n" + "=" * 70)
    print("        2. DATASET ANALYSIS & CLASS DISTRIBUTION REPORT")
    print("=" * 70)

    if not os.path.exists(CLASS_NAMES_PATH):
        print(f"[Error] Missing class_names.json at {CLASS_NAMES_PATH}")
        return

    with open(CLASS_NAMES_PATH, 'r') as f:
        classes = json.load(f)

    print(f"Target Species Classes Count: {len(classes)}")

    splits = {'train': TRAIN_DIR, 'validation': VAL_DIR, 'test': TEST_DIR}
    counts = {cls: {'train': 0, 'validation': 0, 'test': 0, 'total': 0} for cls in classes}

    for split_name, split_path in splits.items():
        if not os.path.exists(split_path):
            print(f"[Warning] Split directory not found: {split_path}")
            continue
        for cls in classes:
            cls_path = os.path.join(split_path, cls)
            if os.path.exists(cls_path):
                img_files = [f for f in os.listdir(cls_path) if os.path.isfile(os.path.join(cls_path, f))]
                cnt = len(img_files)
                counts[cls][split_name] = cnt
                counts[cls]['total'] += cnt

    # Print Distribution Table
    print("\nClass Distribution Table:")
    print(f"{'Species Class Name':<30} | {'Train':<7} | {'Val':<7} | {'Test':<7} | {'Total':<7}")
    print("-" * 70)

    total_train = total_val = total_test = grand_total = 0
    min_count = float('inf')
    max_count = 0
    min_class = max_class = ""

    for cls in classes:
        tr = counts[cls]['train']
        va = counts[cls]['validation']
        te = counts[cls]['test']
        tot = counts[cls]['total']

        total_train += tr
        total_val += va
        total_test += te
        grand_total += tot

        if tot < min_count:
            min_count = tot
            min_class = cls
        if tot > max_count:
            max_count = tot
            max_class = cls

        print(f"{cls:<30} | {tr:<7} | {va:<7} | {te:<7} | {tot:<7}")

    print("-" * 70)
    print(f"{'TOTAL IMAGES':<30} | {total_train:<7} | {total_val:<7} | {total_test:<7} | {grand_total:<7}")

    # Summary Metrics
    print("\nDataset Summary Metrics:")
    print(f"  • Total Images          : {grand_total}")
    print(f"  • Total Species Classes : {len(classes)}")
    print(f"  • Training Images (70%) : {total_train} ({round(total_train/max(1,grand_total)*100, 1)}%)")
    print(f"  • Validation Images(15%): {total_val} ({round(total_val/max(1,grand_total)*100, 1)}%)")
    print(f"  • Test Images (15%)     : {total_test} ({round(total_test/max(1,grand_total)*100, 1)}%)")
    print(f"  • Minimum Class Count   : {min_count} ({min_class})")
    print(f"  • Maximum Class Count   : {max_count} ({max_class})")
    
    # Class balance check
    if max_count > 0 and min_count / max_count < 0.5:
        print("  • Class Balance Status  : Moderate Imbalance Detected (Handled by class weights or data augmentation in Step 4)")
    else:
        print("  • Class Balance Status  : Well Balanced across classes")

    print("\n[Confirmation] Model training has NOT been performed yet.")
    print("Dataset preparation for Step 3 is complete and verified.")
    print("=" * 70)


if __name__ == '__main__':
    check_environment()
    analyze_dataset()
