# LeafScan — Plant Leaf Dataset Documentation (Step 3)

## 📌 Dataset Overview

- **Dataset Name**: PlantVillage Benchmark Subset (Academic Selection)
- **Source**: PlantVillage Project (Penn State University & EPFL, open-access dataset)
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Purpose**: Academic demonstration of Transfer Learning fine-tuning for plant leaf species identification.

---

## 🍃 Target Species Classes (10 Classes)

| Index | Class Folder Name | Plant Crop | Condition |
|:---:|:---|:---|:---|
| 0 | `Apple___Apple_scab` | Apple | Diseased (Apple Scab) |
| 1 | `Apple___Black_rot` | Apple | Diseased (Black Rot) |
| 2 | `Apple___healthy` | Apple | Healthy |
| 3 | `Cherry___healthy` | Cherry | Healthy |
| 4 | `Corn___Common_rust` | Corn / Maize | Diseased (Common Rust) |
| 5 | `Corn___healthy` | Corn / Maize | Healthy |
| 6 | `Grape___Black_rot` | Grape | Diseased (Black Rot) |
| 7 | `Grape___healthy` | Grape | Healthy |
| 8 | `Peach___Bacterial_spot` | Peach | Diseased (Bacterial Spot) |
| 9 | `Potato___Late_blight` | Potato | Diseased (Late Blight) |

---

## 📊 Dataset Split Strategy

The dataset is partitioned into three disjoint subsets using a reproducible random seed (`42`) to prevent data leakage:

- **Training Set (`dataset/train/`)**: **70%** — Used by MobileNetV2 model during step 4 training.
- **Validation Set (`dataset/validation/`)**: **15%** — Used for hyperparameter monitoring and early stopping evaluation during training.
- **Testing Set (`dataset/test/`)**: **15%** — Held out exclusively for final model accuracy evaluation.

```
dataset/
├── raw/                      # Raw ingestion files (ignored by Git)
├── train/                    # 70% Training set (ignored by Git)
├── validation/               # 15% Validation set (ignored by Git)
├── test/                     # 15% Test set (ignored by Git)
└── README.md
```

---

## 🛠️ Data Preparation & Analysis Scripts

1. **Dataset Ingestion & Split Script**:
   ```bash
   python scripts/prepare_dataset.py
   ```
   *Validates image integrity via Pillow, filters corrupted files, and executes the 70/15/15 split.*

2. **Dataset Metrics & Environment Inspection Script**:
   ```bash
   python scripts/analyze_dataset.py
   ```
   *Displays per-class distribution tables, class balance metrics, and hardware accelerator status.*

---

## 🎓 Academic Viva Defense Notes

1. **Why 10 classes?**: Focuses on high-impact agricultural crops while keeping parameters lightweight (~14 MB MobileNetV2 weight file) for fast mobile/web inference.
2. **Preventing Data Leakage**: Test images are isolated before training begins and are never seen by the model during loss optimization.
