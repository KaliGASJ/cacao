from ultralytics import YOLO
import torch

def entrenar_modelo():
    """
    Realiza fine-tuning de YOLOv11s para clasificación.
    """
    # Verificación de hardware (GPU o CPU fallback)
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo activo: {device}")

    # Inicializar modelo
    modelo = YOLO("yolo11s-cls.pt")

    # Iniciar entrenamiento con Data Augmentation y Early Stopping
    modelo.train(
        data="dataset_cacao",
        epochs=100,
        imgsz=224,
        batch=16,
        device=device,
        project="resultados_cacao",
        name="modelo_refinado",
        plots=True,
        save=True,
        patience=15,
        optimizer="AdamW",
        lr0=0.001,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=15.0,
        fliplr=0.5
    )
    
    print("Entrenamiento completado.")

if __name__ == '__main__':
    entrenar_modelo()
