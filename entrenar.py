"""
entrenar.py
-----------
Fine-tuning de YOLOv11m-cls para clasificacion binaria sano/moniliasis.
Optimizado para RTX 5050 (8GB VRAM laptop).

Calibracion de hiperparametros para estabilidad y precision maxima:
  - Augmentacion moderada: evita oscilaciones de val accuracy
  - Label smoothing 0.10: targets 0.05/0.95 (no 0.1/0.9 que limitaba confianza)
  - Dropout 0.35: regulariza sin sobrecastigar
  - Evaluacion final en test holdout (nunca visto en entrenamiento)

Uso:
    python3 entrenar.py
"""

from pathlib import Path
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
        trainer.model.loss = nn.CrossEntropyLoss(label_smoothing=0.10)
        frozen = sum(1 for p in trainer.model.parameters() if not p.requires_grad)
        total  = sum(1 for _ in trainer.model.parameters())
        print(f"[INFO] Label Smoothing 0.10 activo (targets 0.05/0.95)")
        print(f"[INFO] Parametros: {total} total | {frozen} congelados | {total-frozen} entrenables")

    modelo.add_callback("on_train_start", on_train_start)

    resultados = modelo.train(
        data="dataset_cacao",
        epochs=150,
        imgsz=384,
        batch=28,
        device=device,
        exist_ok=True,
        workers=4,

        project="resultados_cacao",
        name="modelo_refinado",
        plots=True,
        save=True,

        freeze=0,

        # ── Optimizador ──
        optimizer="AdamW",
        lr0=0.0003,
        lrf=0.005,
        weight_decay=0.02,
        warmup_epochs=5,
        warmup_momentum=0.5,
        cos_lr=True,

        patience=30,

        # ── Regularizacion calibrada ──
        dropout=0.35,

        # ── Augmentation moderada (reduce oscilaciones) ──
        hsv_h=0.02,
        hsv_s=0.6,
        hsv_v=0.4,
        degrees=20.0,
        translate=0.15,
        scale=0.40,
        fliplr=0.5,
        flipud=0.0,
        perspective=0.0005,
        erasing=0.15,
        mixup=0.10,
        auto_augment="randaugment",
    )

    # ── Evaluacion VAL + TEST ──
    mejor = YOLO(resultados.save_dir / "weights" / "best.pt")

    print("\n" + "=" * 60)
    print("EVALUACION EN VAL (usado para early stopping)")
    print("=" * 60)
    m_val = mejor.val(data="dataset_cacao", device=device, split="val")
    print(f"\nTop-1 Accuracy (VAL):  {m_val.top1:.4f}")
    print(f"Top-5 Accuracy (VAL):  {m_val.top5:.4f}")

    print("\n" + "=" * 60)
    print("EVALUACION EN TEST HOLDOUT")
    print("=" * 60)
    test_dir = Path("dataset_cacao") / "test"
    if test_dir.exists():
        m_test = mejor.val(data="dataset_cacao", device=device, split="test")
        print(f"\nTop-1 Accuracy (TEST): {m_test.top1:.4f}")
        print(f"Top-5 Accuracy (TEST): {m_test.top5:.4f}")
        gap = abs(m_val.top1 - m_test.top1)
        if gap > 0.05:
            print(f"\n[AVISO] Gap val/test = {gap:.2%} — posible sobreajuste al val.")
        else:
            print(f"\n[OK] Gap val/test = {gap:.2%} — dentro del margen esperado.")
    else:
        print("[INFO] No se encontro dataset_cacao/test/ — solo se evaluo val.")

    print(f"\nMatriz de confusion: {resultados.save_dir}/confusion_matrix.png")
    print("=" * 60)


if __name__ == "__main__":
    entrenar_modelo()
