# 🍫 Cocoa Quality & Disease Classifier (Moniliasis)

## 📌 Project Overview

This Capstone project implements a **Computer Vision binary classifier** powered by the **YOLOv11** architecture, designed to distinguish healthy cocoa pods from those infected by *Moniliophthora roreri* (Frosty Pod Rot / Moniliasis).

The system includes a complete end-to-end pipeline: automated dataset preparation, GPU-accelerated model training with Data Augmentation and Early Stopping, and an interactive **Streamlit web dashboard** for real-time inference and metric visualization.

*🇪🇸 Para la versión en español, consulta [README_es.md](README_es.md).*

---

## 🚀 Features

- **Automated Dataset Pipeline:** Drop your photos into `raw/` and `preparar_datos.py` handles shuffling and 80/20 train/val splitting automatically.
- **Optimized Training:** Fine-tuning with `YOLOv11s-cls`, `AdamW` optimizer, and on-the-fly Data Augmentation (HSV shifts, rotation, horizontal flip).
- **Overfitting Prevention:** Built-in `EarlyStopping` (patience=15) ensures the model generalizes to unseen images.
- **Hardware Flexibility:** Automatically uses NVIDIA CUDA GPU if available, with seamless CPU fallback.
- **Interactive Dashboard:** A Streamlit-based web interface to upload images, get instant predictions, and visualize training metrics.

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/KaliGASJ/cacao.git
cd cacao
```

### 2. Create a virtual environment

```bash
python3 -m venv cacao_env #On Windows the command is python -m venv cacao_env
source cacao_env/bin/activate #On Windows the command is cacao_env\Scripts\activate
```

### 3. Install core dependencies

```bash
pip install -r requirements.txt
```

### 4. (Optional) CUDA Support for NVIDIA GPUs (e.g., RTX 5050)

If you have a compatible NVIDIA GPU, install PyTorch with GPU acceleration (Nightly for CUDA 13.0):

```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu130
```

To verify that your GPU was detected correctly, run:

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

---

## 🏃‍♂️ Usage

### Step 1: Prepare the dataset

This script shuffles the images from `raw/` and automatically distributes the dataset (80% Training and 20% Validation).

```bash
python3 preparar_datos.py #On Windows the command is python preparar_datos.py
```

> Creates the `dataset_cacao/` directory with `train/` and `val/` splits ready for YOLO.

### Step 2: Train the model

```bash
python3 entrenar.py #On Windows the command is python entrenar.py
```

> Trains the classifier and saves the best weights, confusion matrices, and loss curves to `runs/classify/resultados_cacao/modelo_refinado/`.

### Step 3: Launch the Dashboard

```bash
streamlit run app.py
```

> Opens an interactive web application at `http://localhost:8501` where you can upload cocoa pod images for instant classification and view training metrics.

---

## 📂 Repository Structure

| File / Folder | Description |
| --- | --- |
| `raw/` | Original photographs organized by class (`sano/`, `moniliasis/`). |
| `preparar_datos.py` | Splits raw images into train/val sets for YOLO. |
| `entrenar.py` | Configures and launches the YOLOv11 fine-tuning pipeline. |
| `app.py` | Streamlit dashboard for inference and metric visualization. |
| `requirements.txt` | Project dependencies (Ultralytics, Streamlit, Pillow). |
| `.gitignore` | Excludes heavy model weights, caches, and generated data. |

*Developed as a Capstone Project — March 2026.*
