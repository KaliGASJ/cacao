"""
entrenar.py
───────────
Ejecuta el fine-tuning de un modelo YOLOv11 Small (clasificación)
sobre el dataset generado por 'preparar_datos.py'.

Incluye:
  - Detección automática de GPU (NVIDIA CUDA) con fallback a CPU.
  - Data Augmentation en memoria (HSV, rotación, flip horizontal).
  - Early Stopping con paciencia de 15 épocas para prevenir sobreajuste.

Uso:
    python3 entrenar.py
"""

from ultralytics import YOLO
import torch


def entrenar_modelo():
    """
    Carga el modelo pre-entrenado YOLOv11s-cls y realiza fine-tuning
    con los datos ubicados en 'dataset_cacao/'. Los resultados
    (pesos, gráficas, matriz de confusión) se guardan automáticamente
    en 'runs/classify/resultados_cacao/modelo_refinado/'.
    """
    # Seleccionar dispositivo: GPU (índice 0) si está disponible, si no CPU
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo activo: {device}")

    # Cargar modelo pre-entrenado (Small Classifier)
    modelo = YOLO("yolo11s-cls.pt")

    # Entrenar con hiperparámetros optimizados
    modelo.train(
        data="dataset_cacao",         # Ruta al dataset (train/ y val/)
        epochs=100,                   # Máximo de épocas (Early Stopping puede detenerlo antes)
        imgsz=224,                    # Resolución de entrada (px)
        batch=16,                     # Tamaño de lote por iteración
        device=device,                # GPU o CPU según disponibilidad
        #optimizacion de resultados
        project="resultados_cacao",   # Carpeta padre para guardar resultados
        name="modelo_refinado",       # Nombre del experimento
        plots=True,                   # Generar gráficas de métricas automáticamente
        save=True,                    # Guardar pesos best.pt y last.pt
        #prevencion y optimizacion
        patience=15,                  # Épocas sin mejora antes de detener (Early Stopping)
        optimizer="AdamW",            # Optimizador con decaimiento de pesos
        lr0=0.001,                    # Tasa de aprendizaje inicial
        # ── Data Augmentation (alteraciones en tiempo real) ──
        hsv_h=0.015,                  # Variación de tono (Hue)
        hsv_s=0.7,                    # Variación de saturación
        hsv_v=0.4,                    # Variación de brillo (Value)
        degrees=15.0,                 # Rotación aleatoria (±15°)
        fliplr=0.5,                   # Probabilidad de volteo horizontal (50 %)
    )

    print("Entrenamiento completado.")


if __name__ == "__main__":
    entrenar_modelo()
