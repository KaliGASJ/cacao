# 🍫 Cocoa Quality & Disease Classifier (Moniliasis)

## 📌 Project Overview
This Capstone project implements a highly optimized **Computer Vision classifier** based on the state-of-the-art **YOLOv11** architecture. The model is specifically fine-tuned to detect and classify the presence of *Moniliophthora roreri* (Frosty Pod Rot or Moniliasis) in cocoa pods.

The project achieves high-performance classification by leveraging Data Augmentation and Early Stopping mechanisms, resulting in a **100% top-1 accuracy** on the validation set.

*🇪🇸 For the Spanish version of this document, please refer to [README_es.md](README_es.md).*

---

## 🚀 Features
- **Raw Data Processing Automation:** The `preparar_datos.py` script automatically shuffles and splits raw images into an 80/20 train/validation ratio, building the directory structure required by Ultralytics on-the-fly.
- **Advanced Fine-Tuning:** The `entrenar.py` script executes an intensive training pipeline using `yolo11s-cls.pt` (Small Classifier), optimized with `AdamW` and aggressive on-the-fly Image Augmentations.
- **Overfitting Prevention:** Includes an `EarlyStopping` criteria (`patience=15`) to ensure the model generalizes effectively.
- **Hardware Acceleration / Fallback:** Strictly requests CUDA bindings if an NVIDIA GPU is present. Features built-in automatic CPU fallback for standard environments.

---

## ⚙️ Environment Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/KaliGASJ/cacao.git
cd cacao
```

### 2. Prepare the Virtual Environment (Recommended)
```bash
python3 -m venv cacao_env
source cacao_env/bin/activate  # On Windows use: cacao_env\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
> **Note on CUDA (GPU Acceleration):** `requirements.txt` installs standard PyTorch. If you intend to run this on a compatible NVIDIA GPU (Compute Capability 12.0+), follow the instructions inside the `requirements.txt` file to fetch the correct PyTorch Nightly build.

---

## 🏃‍♂️ Execution Pipeline

To run the full end-to-end pipeline, follow these exact steps from your terminal:

**Step 1: Data Preparation**
```bash
python3 preparar_datos.py
```
*Expected Output: "Estructura lista en 'dataset_cacao' para entrenar con YOLO."*

**Step 2: Model Training & Evaluation**
```bash
python3 entrenar.py
```
*Expected Output: After epochs are completed, results (including loss graphs and confusion matrix) will be automatically generated and exported to `runs/classify/resultados_cacao/modelo_refinado/`.*

---

## 📂 Repository Structure
- `/raw`: Contains the raw dataset (40 original baseline images categorized in `sano/` and `moniliasis/`).
- `preparar_datos.py`: Helper script to format the dataset.
- `entrenar.py`: Core logic for YOLOv11 fine-tuning.
- `.gitignore`: Secures heavy `.pt` files, caches, and user-generated metrics from being tracked.

*Developed as a Capstone Project - March 2026.*
