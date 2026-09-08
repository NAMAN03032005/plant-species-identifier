"""
LeafScan - Dataset Preparation Script (Step 3)
================================================
This script prepares the plant leaf species dataset for Transfer Learning model training:
  1. Loads target species classes from model/class_names.json.
  2. Scans raw images in dataset/raw/.
  3. Performs image integrity checks (Pillow verification, valid format & size).
  4. Ignores corrupted or non-image files.
  5. Splits images into 70% Train, 15% Validation, and 15% Test.
  6. Uses a fixed random seed (42) to ensure zero data leakage and reproducible splits.

Note: This script performs DATASET PREPARATION ONLY.
Model training (MobileNetV2) takes place in Step 4.
"""

import os
import sys
import shutil
import random
import json
from PIL import Image

# Fixed random seed for reproducible train/val/test splits
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Split Ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Supported Image Extensions
SUPPORTED_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.webp')

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
RAW_DIR = os.path.join(DATASET_DIR, 'raw')
TRAIN_DIR = os.path.join(DATASET_DIR, 'train')
VAL_DIR = os.path.join(DATASET_DIR, 'validation')
TEST_DIR = os.path.join(DATASET_DIR, 'test')
CLASS_NAMES_PATH = os.path.join(BASE_DIR, 'model', 'class_names.json')


def load_class_names():
    """Load target species class names from model/class_names.json."""
    if not os.path.exists(CLASS_NAMES_PATH):
        print(f"[Error] Class names file not found at: {CLASS_NAMES_PATH}")
        sys.exit(1)
    with open(CLASS_NAMES_PATH, 'r') as f:
        return json.load(f)


def verify_image(file_path):
    """
    Verify image file integrity using Pillow.
    Returns True if file is valid, non-corrupted image; False otherwise.
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return False
    try:
        with Image.open(file_path) as img:
            img.verify()  # Verify file structure
        # Re-open to verify readable pixels
        with Image.open(file_path) as img:
            img.load()
        return True
    except Exception:
        return False


def generate_sample_dataset_if_empty(classes):
    """
    If dataset/raw/ is empty, generate sample valid RGB leaf images 
    for each class to ensure out-of-the-box pipeline demonstration.
    """
    has_raw_images = False
    for c in classes:
        cdir = os.path.join(RAW_DIR, c)
        if os.path.exists(cdir) and len(os.listdir(cdir)) > 0:
            has_raw_images = True
            break

    if not has_raw_images:
        print("[Info] No raw images found in dataset/raw/. Generating initial sample leaf dataset...")
        for idx, cls in enumerate(classes):
            cls_dir = os.path.join(RAW_DIR, cls)
            os.makedirs(cls_dir, exist_ok=True)
            # Create 40 sample leaf-colored images per class with unique random noise
            for i in range(1, 41):
                img_path = os.path.join(cls_dir, f"{cls}_sample_{i:03d}.jpg")
                # Unique random seed per sample to guarantee unique file hash
                np.random.seed(idx * 1000 + i)
                base_color = np.array([25 + idx * 15, 120 + (i * 3) % 40, 45 + (idx * 10) % 50], dtype=np.uint8)
                noise = np.random.randint(-15, 15, (224, 224, 3), dtype=np.int16)
                img_array = np.clip(base_color + noise, 0, 255).astype(np.uint8)
                img = Image.fromarray(img_array, 'RGB')
                img.save(img_path, 'JPEG', quality=95)
        print("[Success] Sample leaf dataset created across 10 classes in dataset/raw/.")


def prepare_dataset():
    """Main dataset preparation workflow."""
    print("=" * 65)
    print("      LeafScan - Step 3: Dataset Preparation Pipeline")
    print("=" * 65)

    classes = load_class_names()
    print(f"Target Species Classes ({len(classes)}):")
    for idx, name in enumerate(classes, 1):
        print(f"  {idx:02d}. {name}")
    print("-" * 65)

    # Ensure raw directory sample data exists
    generate_sample_dataset_if_empty(classes)

    # Reset output directories
    for split_dir in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        if os.path.exists(split_dir):
            shutil.rmtree(split_dir)
        os.makedirs(split_dir, exist_ok=True)

    stats = {cls: {'raw': 0, 'train': 0, 'val': 0, 'test': 0, 'corrupted': 0} for cls in classes}

    for cls in classes:
        cls_raw_dir = os.path.join(RAW_DIR, cls)
        if not os.path.exists(cls_raw_dir):
            print(f"[Warning] Class directory missing: {cls_raw_dir}")
            continue

        raw_files = [os.path.join(cls_raw_dir, f) for f in os.listdir(cls_raw_dir)]
        valid_files = []

        for fpath in raw_files:
            if os.path.isfile(fpath):
                if verify_image(fpath):
                    valid_files.append(fpath)
                else:
                    stats[cls]['corrupted'] += 1

        stats[cls]['raw'] = len(valid_files)

        if not valid_files:
            print(f"[Warning] No valid images found for class: {cls}")
            continue

        # Shuffle deterministically with fixed seed
        random.shuffle(valid_files)

        total = len(valid_files)
        train_end = int(total * TRAIN_RATIO)
        val_end = train_end + int(total * VAL_RATIO)

        train_files = valid_files[:train_end]
        val_files = valid_files[train_end:val_end]
        test_files = valid_files[val_end:]

        stats[cls]['train'] = len(train_files)
        stats[cls]['val'] = len(val_files)
        stats[cls]['test'] = len(test_files)

        # Copy files to respective split subdirectories
        for split_name, file_list in [('train', train_files), ('validation', val_files), ('test', test_files)]:
            split_cls_dir = os.path.join(DATASET_DIR, split_name, cls)
            os.makedirs(split_cls_dir, exist_ok=True)
            for src_file in file_list:
                fname = os.path.basename(src_file)
                dst_file = os.path.join(split_cls_dir, fname)
                shutil.copy2(src_file, dst_file)

    print("\nDataset Split Summary:")
    print(f"{'Species Class':<30} | {'Train (70%)':<11} | {'Val (15%)':<10} | {'Test (15%)':<10} | {'Corrupted':<9}")
    print("-" * 80)
    total_train = total_val = total_test = total_corrupt = 0
    for cls in classes:
        tr = stats[cls]['train']
        va = stats[cls]['val']
        te = stats[cls]['test']
        co = stats[cls]['corrupted']
        total_train += tr
        total_val += va
        total_test += te
        total_corrupt += co
        print(f"{cls:<30} | {tr:<11} | {va:<10} | {te:<10} | {co:<9}")

    print("-" * 80)
    print(f"{'TOTAL IMAGES':<30} | {total_train:<11} | {total_val:<10} | {total_test:<10} | {total_corrupt:<9}")
    print(f"\n[Success] Dataset successfully split into dataset/train, dataset/validation, and dataset/test.")
    print("=" * 65)


if __name__ == '__main__':
    prepare_dataset()
