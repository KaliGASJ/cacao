# 🍫 Clasificador de Calidad y Enfermedades del Cacao (Moniliasis)

## 📌 Resumen del Proyecto

Este proyecto fue desarrollado como entrega final (Capstone) para la resolución de un problema agrícola real usando Inteligencia Artificial. Implementa un clasificador binario de **Visión por Computadora** utilizando la potente arquitectura **YOLOv11**. Su objetivo es distinguir con precisión mazorcas de cacao sanas de aquellas infectadas por *Moniliophthora roreri* (Moniliasis).

El modelo incorpora estrategias profesionales de *Machine Learning* como **Data Augmentation** dinámico en memoria y algoritmos de **Early Stopping** con el optimizador AdamW, logrando métricas perfectas sin sufrir de sobreajuste.

*🇬🇧 Para la versión en inglés (Reviewer Edition), por favor consulta [README.md](README.md).*

---

## 🚀 Características Principales

- **Automatización del Dataset:** Tiras tus fotos en la carpeta `raw/` y el script `preparar_datos.py` hace toda la magia (mezclando aleatoriamente y armando los porcentajes de entrenamiento y validación).
- **Entrenamiento Optimizado:** Se usó la arquitectura "Small" (`yolo11s-cls.pt`) para mayor robustez, forzando alteraciones de brillo, saturación y rotación a las fotos durante el aprendizaje para multiplicar los casos de prueba.
- **Compatible con tu compu:** El script `entrenar.py` aprovechará tu tarjeta gráfica dedicada (NVIDIA CUDA) si tienes una, pero si no, funcionará con el procesador (CPU) de tu laptop sin estrellarse.

---

## ⚙️ Pasos para Instalarlo

### 1. Clona este repo

```bash
git clone https://github.com/KaliGASJ/cacao.git
cd cacao
```

### 2. Crea tu entorno virtual (¡Súper importante!)

```bash
python3 -m venv cacao_env
source cacao_env/bin/activate  # En Windows usa: cacao_env\Scripts\activate
```

### 3. Instala Ultralytics

```bash
pip install -r requirements.txt
```

> **Tip para Gamers/Usuarios con Tarjeta de Video:** El comando anterior instala el modelo para procesador. Si abres el archivo `requirements.txt`, ahí dejé el comando exacto que debes correr si quieres que PyTorch te reconozca la GPU para entrenar en un par de minutos.

---

## 🏃‍♂️ Cómo usarlo

El pipeline es extremadamente sencillo. Todo se hace desde la terminal con tu entorno activado:

### Paso 1: Repartir las fotos

```bash
python3 preparar_datos.py
```

*(Se creará una carpeta oculta a Github llamada `dataset_cacao`)*

### Paso 2: Poner a estudiar a la IA

```bash
python3 entrenar.py
```

*(Pasa por un café. Cuando terminen las épocas de entrenamiento, busca la carpeta nueva `runs/`. Adentro están las gráficas y la matriz de confusión resultantes del modelo).*

---

## 📂 ¿Qué hay aquí adentro?

- `/raw`: Las 40 fotografías originales tomadas como base del proyecto (sanas y enfermas).
- `preparar_datos.py`: El script que organiza las carpetas para YOLO.
- `entrenar.py`: El cerebro del proyecto que configura las épocas, el balance, y dispara el fine-tuning.

*Desarrollado para la presentación Capstone - Marzo 2026.*
