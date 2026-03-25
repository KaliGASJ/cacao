# CacaoVision — Cocoa Disease Classifier (Moniliasis)

Computer Vision binary classifier powered by **YOLOv11m-cls** that distinguishes healthy cocoa pods from those infected by *Moniliophthora roreri* (Frosty Pod Rot / Moniliasis).

*🇪🇸 Para la versión en español, consulta [README_es.md](README_es.md).*

---

## Features

- **Automated dataset pipeline** — drop photos into `raw/sano/` and `raw/moniliasis/`, the script handles stratified 80/20 split, oversampling, and offline crop augmentation automatically.
- **Strong training setup** — YOLOv11m-cls fine-tuning with AdamW, cosine LR schedule, label smoothing, mixup, and aggressive augmentation (rotation ±45°, HSV, erasing).
- **Asymmetric inference thresholds** — `UMBRAL_SANO = 92%` / `UMBRAL_MONILIASIS = 10%` to minimize false negatives on diseased pods.
- **Hardware flexible** — auto-detects NVIDIA CUDA GPU; falls back to CPU seamlessly.
- **Modern dashboard** — Streamlit UI with real-time predictions, dual confidence bars, and training metric plots.

---

## Installation

### 1. Clone the repository

```batch
git clone https://github.com/KaliGASJ/cacao.git
cd cacao
```

### 2. Create a virtual environment

```batch
python -m venv cacao_env                # Linux / Mac: python3 -m venv cacao_env
cacao_env\Scripts\activate              # Linux / Mac: source cacao_env/bin/activate
```

### 3. Install dependencies

```batch
pip install -r requirements.txt
```

### 4. (Optional) GPU acceleration — NVIDIA CUDA 13.0

```batch
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu130
```

Verify GPU detection:

```batch
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

---

## Usage

Run the three steps **in order**. Each step depends on the output of the previous one.

### Step 1 — Prepare the dataset

Reads `raw/sano/` and `raw/moniliasis/`, applies stratified split, oversampling, and generates crops. Outputs `dataset_cacao/train/` and `dataset_cacao/val/`.

```batch
python preparar_datos.py                # Linux / Mac: python3 preparar_datos.py
```

### Step 2 — Train the model

Fine-tunes YOLOv11m-cls for up to 150 epochs with early stopping (patience = 30). Saves best weights and metric plots to `runs/classify/resultados_cacao/modelo_refinado/`.

```batch
python entrenar.py                      # Linux / Mac: python3 entrenar.py
```

> [!CAUTION]
> **Hardware Warning:** Training `entrenar.py` on a **CPU takes several hours** under intense, continuous load, leading to potential overheating or hardware damage. We strongly recommend using a **GPU with CUDA**, which finishes in **less than 20 minutes** (e.g., RTX 5050 with `imgsz=512, batch=20`).

### Step 3 — Launch the dashboard

```batch
streamlit run app.py
```

Opens at `http://localhost:8501`. Upload one or more cocoa pod images to get an instant SANO / MONILIASIS / INCIERTO diagnosis with confidence bars.

---

## Project Structure

| Path | Description |
| --- | --- |
| `raw/sano/` | Original healthy pod photos. |
| `raw/moniliasis/` | Original diseased pod photos. |
| `preparar_datos.py` | Dataset split, oversampling, and crop generation. |
| `entrenar.py` | YOLOv11m-cls fine-tuning pipeline. |
| `app.py` | Streamlit inference dashboard. |
| `requirements.txt` | Python dependencies. |
| `dataset_cacao/` | *(auto-generated)* Train / val splits for YOLO. |
| `runs/` | *(auto-generated)* Model weights and training plots. |
| `cacao_env/` | *(auto-generated)* Virtual environment — excluded from git. |

---

Capstone Project — Comalcalco, Tabasco · March 2026
