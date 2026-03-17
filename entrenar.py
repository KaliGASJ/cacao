from ultralytics import YOLO
import torch

def entrenar_modelo_refinado():
    """
    Script AVANZADO para realizar fine-tuning del modelo YOLOv11 de clasificación.
    Se utiliza YOLOv11 Small (yolo11s-cls.pt) para mayor precisión.
    Aplica Data Augmentation nativo de Ultralytics y optimización avanzada.
    """
    # 1. Verificamos que CUDA esté disponible para usar tu RTX 5050
    if torch.cuda.is_available():
        print(f"✅ CUDA detectado. Entrenando con: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ Advertencia: No se detectó CUDA. Se usará la CPU.")

    # 2. Cargamos un modelo MÁS INTELIGENTE: YOLOv11 'Small' (S) en lugar del 'Nano' (N)
    print("\nInicializando modelo YOLOv11 Small (Más Robusto)...")
    modelo = YOLO("yolo11s-cls.pt")

    print("\nIniciando entrenamiento REFINADO con Data Augmentation...")
    # 3. Entrenamos el modelo usando técnicas de un Senior ML Engineer
    resultados = modelo.train(
        # ---- CONFIGURACIÓN BÁSICA ----
        data="dataset_cacao",
        epochs=100,            # Subimos a 100 asumiendo que el early-stopping nos protegerá
        imgsz=224,
        batch=16,
        device=0,
        project="resultados_cacao",
        name="modelo_refinado",
        plots=True,
        save=True,
        
        # ---- TÉCNICAS DE REFINAMIENTO (NIVEL SENIOR) ----
        patience=15,           # EARLY STOPPING: Si la precisión no mejora en 15 épocas, se detiene para evitar el sobreajuste (overfitting)
        optimizer="AdamW",     # OPTIMIZADOR: AdamW es el estándar la industria para visión computacional, da curvas de pérdida mucho más estables
        lr0=0.001,             # LEARNING RATE INICIAL: Tasa de aprendizaje moderada
        
        # ---- DATA AUGMENTATION INTENSIVO ----
        # Dado que tienes pocas imágenes, forzamos al modelo a ver variaciones matemáticas (iluminación, giros) de las mismas fotos
        hsv_h=0.015,           # Variación sutil de los tonos de color (matiz)
        hsv_s=0.7,             # Variación agresiva de saturación (simula sol fuerte vs. día nublado)
        hsv_v=0.4,             # Variación de exposición/brillo (simula sombras en la siembra)
        degrees=15.0,          # Rota las fotos hasta 15 grados al azar
        flipud=0.0,            # (Apagado) No damos vuelta al cacao de cabeza, no tiene sentido en la naturaleza
        fliplr=0.5,            # (Prendido 50%) Refleja la foto como espejo de izquierda a derecha
    )

    print("\n🏆 ¡Entrenamiento Avanzado Completado!")
    print("Revisa la carpeta: 'resultados_cacao/modelo_refinado/' para obtener gráficas significativamente mejores.")

if __name__ == '__main__':
    entrenar_modelo_refinado()
