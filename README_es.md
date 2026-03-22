# 🍫 Clasificador de Calidad y Enfermedades del Cacao (Moniliasis)

## 📌 Resumen del Proyecto

Este proyecto fue desarrollado como entrega final (Capstone) para la resolución de un problema agrícola real usando Inteligencia Artificial. Implementa un clasificador binario de **Visión por Computadora** con la arquitectura **YOLOv11**, capaz de distinguir mazorcas de cacao sanas de aquellas infectadas por *Moniliophthora roreri* (Moniliasis). Actualmente alcanza un **98% de precisión** en el conjunto de validación.

El sistema incluye un pipeline completo: preparación automática del dataset, entrenamiento acelerado por GPU con **Data Augmentation** y **Early Stopping**, y un **Dashboard web interactivo** construido con Streamlit para realizar inferencia en tiempo real y visualizar las métricas del modelo.

*🇬🇧 For the English version (Reviewer Edition), please refer to [README.md](README.md).*

---

## 🚀 Características Principales

- **Automatización del Dataset:** Coloca tus fotos en `raw/` y el script `preparar_datos.py` las mezcla y divide automáticamente en 80% entrenamiento / 20% validación.
- **Entrenamiento Optimizado:** Fine-tuning con `YOLOv11s-cls`, optimizador `AdamW`, y Data Augmentation en tiempo real (cambios de brillo, saturación, rotación y volteo horizontal).
- **Prevención de Sobreajuste:** Incluye `EarlyStopping` (paciencia de 15 épocas) para garantizar que el modelo generalice con imágenes nuevas.
- **Compatible con tu compu:** Aprovecha automáticamente tu tarjeta gráfica NVIDIA (CUDA) si tienes una, pero funciona sin problema en CPU.
- **Dashboard Interactivo:** Interfaz web moderna con Streamlit para subir imágenes, obtener predicciones instantáneas y visualizar las métricas de entrenamiento.

---

## ⚙️ Instalación

### 1. Clona este repositorio

```bash
git clone https://github.com/KaliGASJ/cacao.git
cd cacao
```

### 2. Crea tu entorno virtual

```bash
python3 -m venv cacao_env #En windows el comando es python -m venv cacao_env
source cacao_env/bin/activate #En windows el comando es cacao_env\Scripts\activate
```

### 3. Instala las dependencias

```bash
pip install -r requirements.txt
```

> **Tip para usuarios con GPU:** El comando anterior instala PyTorch en modo CPU. Si tienes una tarjeta NVIDIA, revisa las instrucciones dentro de `requirements.txt` para instalar la versión con aceleración CUDA.

---

## 🏃‍♂️ Cómo Usarlo

### Paso 1: Preparar las fotos

```bash
python3 preparar_datos.py #En windows el comando es python preparar_datos.py
```

> Crea la carpeta `dataset_cacao/` con las subcarpetas `train/` y `val/` listas para YOLO.

### Paso 2: Entrenar la IA

```bash
python3 entrenar.py #En windows el comando es python entrenar.py
```

> Entrena el clasificador y guarda los mejores pesos, matrices de confusión y curvas de pérdida en `runs/classify/resultados_cacao/modelo_refinado/`.

### Paso 3: Abrir el Dashboard

```bash
streamlit run app.py
```

> Abre una aplicación web interactiva en `http://localhost:8501` donde puedes arrastrar imágenes de mazorcas y obtener la clasificación instantánea, además de revisar las métricas del modelo.

---

## 📂 Estructura del Repositorio

| Archivo / Carpeta | Descripción |
| --- | --- |
| `raw/` | Fotografías originales organizadas por clase (`sano/`, `moniliasis/`). |
| `preparar_datos.py` | Divide las imágenes en conjuntos de entrenamiento y validación. |
| `entrenar.py` | Configura y ejecuta el fine-tuning de YOLOv11. |
| `app.py` | Dashboard de Streamlit para inferencia y visualización de métricas. |
| `requirements.txt` | Dependencias del proyecto (Ultralytics, Streamlit, Pillow, OpenCV). |
| `.gitignore` | Excluye pesos pesados, cachés y datos generados del repositorio. |

*Desarrollado para la presentación Capstone — Marzo 2026.*
