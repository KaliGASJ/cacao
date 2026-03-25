# CacaoVision — Clasificador de Enfermedades del Cacao (Moniliasis)

Clasificador binario de **Visión por Computadora** con arquitectura **YOLOv11m-cls** que distingue mazorcas de cacao sanas de aquellas infectadas por *Moniliophthora roreri* (Moniliasis).

*🇬🇧 For the English version, see [README.md](README.md).*

---

## Características

- **Pipeline de dataset automatizado** — coloca fotos en `raw/sano/` y `raw/moniliasis/`; el script aplica split estratificado 80/20, oversampling de clase minoritaria y generación de crops offline.
- **Entrenamiento robusto** — fine-tuning de YOLOv11m-cls con AdamW, cosine LR, label smoothing, mixup y augmentación agresiva (rotación ±45°, HSV, erasing).
- **Umbrales asimétricos** — `UMBRAL_SANO = 92%` / `UMBRAL_MONILIASIS = 10%` para minimizar falsos negativos en mazorcas enfermas.
- **Compatible con tu hardware** — detecta GPU NVIDIA (CUDA) automáticamente; funciona en CPU sin cambios.
- **Dashboard moderno** — interfaz Streamlit con predicciones en tiempo real, barras duales de confianza y gráficas de entrenamiento.

---

## Instalación

### 1. Clona el repositorio

```batch
git clone https://github.com/KaliGASJ/cacao.git
cd cacao
```

### 2. Crea el entorno virtual

```batch
python -m venv cacao_env                # Linux / Mac: python3 -m venv cacao_env
cacao_env\Scripts\activate              # Linux / Mac: source cacao_env/bin/activate
```

### 3. Instala las dependencias

```batch
pip install -r requirements.txt
```

### 4. (Opcional) Aceleración GPU — NVIDIA CUDA 13.0

```batch
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu130
```

Verifica que la GPU fue detectada:

```batch
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

---

## Uso

Ejecuta los tres pasos **en orden**. Cada uno depende de la salida del anterior.

### Paso 1 — Preparar el dataset

Lee `raw/sano/` y `raw/moniliasis/`, aplica split estratificado, oversampling y genera crops. Genera `dataset_cacao/train/` y `dataset_cacao/val/`.

```batch
python preparar_datos.py                # Linux / Mac: python3 preparar_datos.py
```

### Paso 2 — Entrenar el modelo

Fine-tuning de YOLOv11m-cls hasta 150 épocas con early stopping (patience = 30). Guarda los mejores pesos y gráficas en `runs/classify/resultados_cacao/modelo_refinado/`.

```batch
python entrenar.py                      # Linux / Mac: python3 entrenar.py
```

> [!CAUTION]
> **Peligro por estrés del hardware:** Ejecutar `entrenar.py` en **CPU tomará varias horas** bajo un nivel de procesamiento al 100% y constante. Esto genera un alto riesgo de sobrecalentamiento que a la larga puede dañar tu equipo. Por seguridad, usar una **GPU con CUDA es estrictamente recomendado** (toma **menos de 20 minutos**, ej. en RTX 5050 con `imgsz=512, batch=20`).

### Paso 3 — Abrir el dashboard

```batch
streamlit run app.py
```

Se abre en `http://localhost:8501`. Sube una o varias imágenes de mazorcas para obtener diagnóstico instantáneo SANO / MONILIASIS / INCIERTO con barras de confianza.

---

## Estructura del Proyecto

| Ruta | Descripción |
| --- | --- |
| `raw/sano/` | Fotos originales de mazorcas sanas. |
| `raw/moniliasis/` | Fotos originales de mazorcas enfermas. |
| `preparar_datos.py` | Split, oversampling y generación de crops. |
| `entrenar.py` | Pipeline de fine-tuning YOLOv11m-cls. |
| `app.py` | Dashboard de inferencia con Streamlit. |
| `requirements.txt` | Dependencias de Python. |
| `dataset_cacao/` | *(auto-generado)* Splits train/val para YOLO. |
| `runs/` | *(auto-generado)* Pesos del modelo y gráficas de entrenamiento. |
| `cacao_env/` | *(auto-generado)* Entorno virtual — excluido del repositorio. |

---

Proyecto Capstone — Comalcalco, Tabasco · Marzo 2026
