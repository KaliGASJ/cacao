"""
entrenar.py
-----------
Fine-tuning de YOLOv11m-cls para clasificacion binaria de Moniliasis.
Optimizado para RTX 5050 (8GB VRAM laptop).

Cambios vs version anterior:
  - LR reducido para evitar overfitting rapido (peak en epoca 29)
  - patience reducido: el modelo no mejora despues del peak
  - label_smoothing aumentado a 0.2: penaliza mas la sobreconfianza
  - batch aumentado a 16: gradientes mas estables (VRAM ok ~2.3GB)
  - freeze=0: sin congelar, el modelo aprende texturas especificas de cacao

Uso:
    python3 entrenar.py
"""

from ultralytics import YOLO
import torch
import torch.nn as nn


def entrenar_modelo():
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo: {'CUDA (GPU 0)' if device == 0 else 'CPU'}")

    if device == 0:
        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / (1024 ** 3)
        print(f"GPU: {props.name} ({vram_gb:.1f} GB VRAM)")

    modelo = YOLO("yolo11m-cls.pt")

    def on_train_start(trainer):
        # Label smoothing mas alto: penaliza sobreconfianza (el problema principal)
        trainer.model.loss = nn.CrossEntropyLoss(label_smoothing=0.2)
        frozen = sum(1 for p in trainer.model.parameters() if not p.requires_grad)
        total  = sum(1 for _  in trainer.model.parameters())
        print(f"[INFO] Label Smoothing 0.20 activo")
        print(f"[INFO] Parametros: {total} total | {frozen} congelados | {total-frozen} entrenables")

    modelo.add_callback("on_train_start", on_train_start)

    resultados = modelo.train(
        data="dataset_cacao",
        epochs=150,
        imgsz=384,                    # 384 captura textura moniliasis, ~55% menos computo que 512
        batch=28,                     # Mas batch con VRAM liberada (~3.5GB estimado)
        device=device,
        exist_ok=True,
        workers=4,                    # Mas workers ahora que hay margen de RAM

        project="resultados_cacao",
        name="modelo_refinado",
        plots=True,
        save=True,

        # Sin congelar: el modelo necesita aprender texturas especificas de cacao
        # (moniliasis temprana tiene patrones que ImageNet no cubre)
        freeze=0,

        # ── Optimizador ──
        # LR mas bajo: el pico anterior en ep.29 con LR 0.0008 causaba
        # que el modelo memorizara en vez de generalizar
        optimizer="AdamW",
        lr0=0.0003,
        lrf=0.005,
        weight_decay=0.02,            # Mayor L2: frena memorizar casos faciles
        warmup_epochs=5,              # Warmup corto: mas epocas reales de aprendizaje
        warmup_momentum=0.5,
        cos_lr=True,

        # Patience amplio: warmup=5 + modelo necesita tiempo para generalizar
        # Con 150 epocas y lr bajo, el plateau real ocurre mucho despues
        patience=30,

        # ── Regularizacion ──
        dropout=0.45,                 # Mayor dropout: fuerza al modelo a no depender
                                      # de features individuales (generaliza mejor)

        # ── Augmentation ──
        hsv_h=0.02,
        hsv_s=0.8,
        hsv_v=0.6,
        degrees=45.0,
        translate=0.25,
        scale=0.55,
        fliplr=0.5,
        flipud=0.3,
        perspective=0.001,
        erasing=0.25,                 # Reducido: 0.4 cubria demasiada textura de moniliasis
        mixup=0.25,                   # Mas mixup: mezcla agresiva de clases
        auto_augment="randaugment",
    )

    # ── Evaluacion ──
    print("\n" + "=" * 60)
    print("EVALUACION EN CONJUNTO DE VALIDACION")
    print("=" * 60)

    mejor = YOLO(resultados.save_dir / "weights" / "best.pt")
    m = mejor.val(data="dataset_cacao", device=device, split="val")

    print(f"\nTop-1 Accuracy: {m.top1:.4f}")
    print(f"Top-5 Accuracy: {m.top5:.4f}")
    print(f"\nMatriz de confusion: {resultados.save_dir}/confusion_matrix.png")
    print("=" * 60)


if __name__ == "__main__":
    entrenar_modelo()
