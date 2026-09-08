# Raw Dataset Directory

This directory holds original, un-split raw image files collected from the academic **PlantVillage Dataset**.

## Raw Data Ingestion Guidelines
1. Place class folders inside `dataset/raw/` (e.g. `dataset/raw/Apple___healthy/`).
2. Run `python scripts/prepare_dataset.py` to validate images and perform a 70% train / 15% validation / 15% test split into `dataset/train/`, `dataset/validation/`, and `dataset/test/`.

> **Note**: Raw images are kept out of version control via `.gitignore`.
