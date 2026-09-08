"""
LeafScan - MobileNetV2 Transfer Learning Model Training Script (Step 4)
========================================================================
This script trains a Deep Learning plant leaf species classifier using Transfer Learning:
  1. Base Architecture: MobileNetV2 pretrained on ImageNet (include_top=False).
  2. Input Shape: 224 x 224 x 3.
  3. Data Augmentation: Training dataset ONLY (Flip, Rotation, Zoom, Translation).
  4. Phase 1 (Feature Extraction): Base layers frozen, Adam (lr=1e-3).
  5. Phase 2 (Fine-Tuning): Unfreeze last 30 layers of MobileNetV2, Adam (lr=1e-5).
  6. Model Checkpointing: Saves best validation model to model/leafscan_mobilenetv2.keras.
  7. Evaluation: Evaluates best model on unseen TEST dataset (Accuracy, Precision, Recall, F1).
  8. Artifact Generation:
     - model/leafscan_mobilenetv2.keras (Model weight file)
     - model/model_metadata.json (Training configuration & metadata)
     - model/training_history.json (Epoch loss and accuracy logs)
     - model/classification_report.txt (Test set precision/recall/f1)
     - model/confusion_matrix.png (Test set confusion matrix plot)
     - model/training_accuracy.png (Accuracy curve plot)
     - model/training_loss.png (Loss curve plot)
"""

import os
import sys
import json
import time
import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server script plotting
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

# Fix random seeds for reproducibility
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

# Hyperparameters
IMG_SIZE = (224, 224)
INPUT_SHAPE = (224, 224, 3)
BATCH_SIZE = 16
PHASE1_MAX_EPOCHS = 12
PHASE2_MAX_EPOCHS = 8
PHASE1_LR = 1e-3
PHASE2_LR = 1e-5

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
TRAIN_DIR = os.path.join(DATASET_DIR, 'train')
VAL_DIR = os.path.join(DATASET_DIR, 'validation')
TEST_DIR = os.path.join(DATASET_DIR, 'test')
MODEL_DIR = os.path.join(BASE_DIR, 'model')

MODEL_SAVE_PATH = os.path.join(MODEL_DIR, 'leafscan_mobilenetv2.keras')
METADATA_PATH = os.path.join(MODEL_DIR, 'model_metadata.json')
HISTORY_PATH = os.path.join(MODEL_DIR, 'training_history.json')
REPORT_PATH = os.path.join(MODEL_DIR, 'classification_report.txt')
CM_PLOT_PATH = os.path.join(MODEL_DIR, 'confusion_matrix.png')
ACC_PLOT_PATH = os.path.join(MODEL_DIR, 'training_accuracy.png')
LOSS_PLOT_PATH = os.path.join(MODEL_DIR, 'training_loss.png')
CLASS_NAMES_PATH = os.path.join(MODEL_DIR, 'class_names.json')


def print_env_info():
    """Detect and print Python, TensorFlow, and hardware acceleration status."""
    print("=" * 70)
    print("      LeafScan - Step 4: MobileNetV2 Model Training Pipeline")
    print("=" * 70)
    print(f"Python Version    : {sys.version.split()[0]}")
    print(f"TensorFlow        : v{tf.__version__}")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"Hardware GPU(s)   : {len(gpus)} GPU(s) detected ({gpus[0].name})")
    else:
        print("Hardware CPU/GPU  : CPU Mode (Configured with batch_size=16 for low memory footprint)")
    print("-" * 70)


def load_classes():
    """Load target class labels from model/class_names.json."""
    if not os.path.exists(CLASS_NAMES_PATH):
        print(f"[Error] Missing class label file: {CLASS_NAMES_PATH}")
        sys.exit(1)
    with open(CLASS_NAMES_PATH, 'r') as f:
        return json.load(f)


def build_data_pipelines(class_names):
    """
    Create TensorFlow tf.data pipelines for Train, Validation, and Test.
    Label mode is categorical (one-hot).
    """
    print("\n[Data Pipeline] Loading image datasets...")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        labels='inferred',
        label_mode='categorical',
        class_names=class_names,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=SEED
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        VAL_DIR,
        labels='inferred',
        label_mode='categorical',
        class_names=class_names,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        labels='inferred',
        label_mode='categorical',
        class_names=class_names,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # Prefetch for performance
    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    test_ds = test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds


def calculate_class_weights(class_names):
    """Calculate class weights based on training dataset sample distribution."""
    counts = []
    for cls in class_names:
        cdir = os.path.join(TRAIN_DIR, cls)
        counts.append(len(os.listdir(cdir)) if os.path.exists(cdir) else 1)
    
    total = sum(counts)
    num_classes = len(class_names)
    class_weights = {}
    print("\nTraining Set Class Balance Inspection:")
    for idx, (cls, count) in enumerate(zip(class_names, counts)):
        # Balanced formula: total / (num_classes * count)
        w = total / (num_classes * max(1, count))
        class_weights[idx] = round(w, 4)
        print(f"  [{idx}] {cls:<30} Count: {count:<4} Weight: {class_weights[idx]}")
    
    return class_weights


def build_model(num_classes):
    """
    Build Transfer Learning Architecture:
    Input -> Data Augmentation -> MobileNetV2 Preprocess -> MobileNetV2 Base -> GlobalAvgPool -> Dropout -> Dense(Softmax)
    """
    # 1. Data Augmentation (Applied ONLY during training)
    data_augmentation = models.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.10),
        layers.RandomZoom(0.10),
        layers.RandomTranslation(0.05, 0.05)
    ], name="data_augmentation")

    # 2. Base Model (Pretrained MobileNetV2 on ImageNet)
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=INPUT_SHAPE,
        include_top=False,
        weights="imagenet"
    )
    
    # Phase 1: Freeze base model
    base_model.trainable = False

    # 3. Model Functional Flow
    inputs = tf.keras.Input(shape=INPUT_SHAPE, name="input_image")
    augmented = data_augmentation(inputs)
    preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(augmented)
    
    # Pass preprocessed image into MobileNetV2 base
    base_features = base_model(preprocessed, training=False)
    
    # Classification Head
    pooled = layers.GlobalAveragePooling2D(name="global_avg_pool")(base_features)
    dropout = layers.Dropout(0.3, name="dropout_layer")(pooled)
    outputs = layers.Dense(num_classes, activation="softmax", name="species_classification")(dropout)

    model = models.Model(inputs=inputs, outputs=outputs, name="LeafScan_MobileNetV2")
    return model, base_model


def plot_and_save_curves(hist1, hist2):
    """Combine Phase 1 and Phase 2 history logs into accuracy and loss plots."""
    acc = hist1.history['accuracy'] + (hist2.history['accuracy'] if hist2 else [])
    val_acc = hist1.history['val_accuracy'] + (hist2.history['val_accuracy'] if hist2 else [])
    loss = hist1.history['loss'] + (hist2.history['loss'] if hist2 else [])
    val_loss = hist1.history['val_loss'] + (hist2.history['val_loss'] if hist2 else [])

    epochs_range = range(1, len(acc) + 1)
    phase1_length = len(hist1.history['accuracy'])

    # 1. Accuracy Plot
    plt.figure(figsize=(9, 5))
    plt.plot(epochs_range, acc, 'o-', label='Training Accuracy', color='#059669', linewidth=2)
    plt.plot(epochs_range, val_acc, 's-', label='Validation Accuracy', color='#0284c7', linewidth=2)
    if hist2:
        plt.axvline(x=phase1_length + 0.5, color='#dc2626', linestyle='--', label='Fine-Tuning Start (Phase 2)')
    plt.title('LeafScan MobileNetV2 - Training & Validation Accuracy', fontsize=13, fontweight='bold')
    plt.xlabel('Epochs', fontsize=11)
    plt.ylabel('Accuracy', fontsize=11)
    plt.legend(loc='lower right')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(ACC_PLOT_PATH, dpi=200)
    plt.close()

    # 2. Loss Plot
    plt.figure(figsize=(9, 5))
    plt.plot(epochs_range, loss, 'o-', label='Training Loss', color='#d97706', linewidth=2)
    plt.plot(epochs_range, val_loss, 's-', label='Validation Loss', color='#9333ea', linewidth=2)
    if hist2:
        plt.axvline(x=phase1_length + 0.5, color='#dc2626', linestyle='--', label='Fine-Tuning Start (Phase 2)')
    plt.title('LeafScan MobileNetV2 - Training & Validation Loss', fontsize=13, fontweight='bold')
    plt.xlabel('Epochs', fontsize=11)
    plt.ylabel('Loss (Categorical Crossentropy)', fontsize=11)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(LOSS_PLOT_PATH, dpi=200)
    plt.close()

    return {
        'accuracy': [float(x) for x in acc],
        'val_accuracy': [float(x) for x in val_acc],
        'loss': [float(x) for x in loss],
        'val_loss': [float(x) for x in val_loss]
    }


def main():
    print_env_info()
    class_names = load_classes()
    num_classes = len(class_names)
    print(f"Loaded {num_classes} species classes from class_names.json.")

    # 1. Pipelines & Weights
    train_ds, val_ds, test_ds = build_data_pipelines(class_names)
    class_weights = calculate_class_weights(class_names)

    # 2. Build Model
    model, base_model = build_model(num_classes)
    model.summary()

    # Callbacks
    checkpoint_cb = callbacks.ModelCheckpoint(
        MODEL_SAVE_PATH,
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    early_stopping_cb = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=4,
        restore_best_weights=True,
        verbose=1
    )
    reduce_lr_cb = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )

    # -------------------------------------------------------------
    # PHASE 1: Feature Extraction (Base Model Frozen)
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  PHASE 1 — FEATURE EXTRACTION (Base Model Frozen, lr=1e-3)")
    print("=" * 70)

    model.compile(
        optimizer=optimizers.Adam(learning_rate=PHASE1_LR),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    start_time = time.time()
    history_phase1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=PHASE1_MAX_EPOCHS,
        class_weight=class_weights,
        callbacks=[checkpoint_cb, early_stopping_cb, reduce_lr_cb]
    )

    # -------------------------------------------------------------
    # PHASE 2: Fine-Tuning (Unfreeze Top 30 MobileNetV2 Layers)
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  PHASE 2 — FINE-TUNING (Unfreezing Top 30 MobileNetV2 Layers, lr=1e-5)")
    print("=" * 70)
    print("Note: Fine-tuning uses a small learning rate (1e-5) to adapt high-level")
    print("ImageNet features to specific plant leaf details without ruining pretrained weights.")

    base_model.trainable = True
    # Freeze all layers except the last 30
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=optimizers.Adam(learning_rate=PHASE2_LR),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    history_phase2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=PHASE2_MAX_EPOCHS,
        class_weight=class_weights,
        callbacks=[checkpoint_cb, early_stopping_cb]
    )

    total_time_min = round((time.time() - start_time) / 60, 2)

    # Save training curves & history log
    combined_history = plot_and_save_curves(history_phase1, history_phase2)
    with open(HISTORY_PATH, 'w') as f:
        json.dump(combined_history, f, indent=2)

    # -------------------------------------------------------------
    # EVALUATION ON UNSEEN TEST DATASET
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  EVALUATING BEST SAVED MODEL ON UNSEEN TEST DATASET")
    print("=" * 70)

    # Reload best saved model
    best_model = models.load_model(MODEL_SAVE_PATH)

    y_true = []
    y_pred = []

    for x_batch, y_batch in test_ds:
        preds = best_model.predict(x_batch, verbose=0)
        y_true.extend(np.argmax(y_batch.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate overall metrics
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    test_loss, test_acc = best_model.evaluate(test_ds, verbose=0)

    print(f"\nFinal Measured Test Set Results:")
    print(f"  • Test Loss     : {test_loss:.4f}")
    print(f"  • Test Accuracy : {test_acc * 100:.2f}%")
    print(f"  • Precision     : {precision:.4f}")
    print(f"  • Recall        : {recall:.4f}")
    print(f"  • F1-Score      : {f1:.4f}")

    # Generate Classification Report
    clr_str = classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0)
    print("\nClassification Report:\n", clr_str)
    
    with open(REPORT_PATH, 'w') as f:
        f.write("LeafScan MobileNetV2 - Test Set Classification Report\n")
        f.write("=" * 55 + "\n")
        f.write(f"Test Accuracy : {test_acc * 100:.2f}%\n")
        f.write(f"Weighted F1   : {f1:.4f}\n")
        f.write("=" * 55 + "\n\n")
        f.write(clr_str)

    # Generate & Plot Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('LeafScan MobileNetV2 - Confusion Matrix (Test Set)', fontsize=13, fontweight='bold')
    plt.xlabel('Predicted Plant Species', fontsize=11)
    plt.ylabel('Actual True Species', fontsize=11)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    plt.savefig(CM_PLOT_PATH, dpi=200)
    plt.close()

    # Save Model Metadata JSON
    metadata = {
        "model_name": "LeafScan MobileNetV2",
        "architecture": "MobileNetV2 Transfer Learning",
        "pretrained_weights": "ImageNet",
        "input_shape": list(INPUT_SHAPE),
        "num_classes": num_classes,
        "class_names": class_names,
        "batch_size": BATCH_SIZE,
        "phase1_max_epochs": PHASE1_MAX_EPOCHS,
        "phase2_max_epochs": PHASE2_MAX_EPOCHS,
        "phase1_epochs_completed": len(history_phase1.history['accuracy']),
        "phase2_epochs_completed": len(history_phase2.history['accuracy']),
        "training_time_minutes": total_time_min,
        "fine_tuning_used": True,
        "unfrozen_layers": 30,
        "test_metrics": {
            "test_loss": round(float(test_loss), 4),
            "test_accuracy": round(float(test_acc), 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1_score": round(float(f1), 4)
        },
        "trained_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(METADATA_PATH, 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 70)
    print(f"[Success] Model training and evaluation complete!")
    print(f"  - Saved Best Model  : {MODEL_SAVE_PATH}")
    print(f"  - Model Metadata    : {METADATA_PATH}")
    print(f"  - Classification Rpt: {REPORT_PATH}")
    print(f"  - Confusion Matrix  : {CM_PLOT_PATH}")
    print(f"  - Accuracy Curve    : {ACC_PLOT_PATH}")
    print(f"  - Loss Curve        : {LOSS_PLOT_PATH}")
    print("=" * 70)


if __name__ == '__main__':
    main()
