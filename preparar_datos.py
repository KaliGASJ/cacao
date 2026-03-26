"""
preparar_datos.py
-----------------
Pipeline de preparacion de datos para clasificacion binaria sano/moniliasis.

Fases:
  1. Escaneo de raw/ con deduplicacion perceptual (average hash)
  2. Split balanceado: test holdout → val → train
  3. Oversampling de clase minoritaria en train
  4. Crops aleatorios offline para romper sesgo de composicion

Uso:
    python3 preparar_datos.py
"""

import shutil
import random
from pathlib import Path
from collections import Counter

import numpy as np
from PIL import Image


# ─── Deduplicacion ────────────────────────────────────────────────────────────

def _ahash(img_path, size=16):
    """Average perceptual hash (256 bits)."""
    try:
        img = Image.open(img_path).convert("L").resize((size, size), Image.LANCZOS)
        pixels = np.array(img, dtype=np.float32).flatten()
        return (pixels > pixels.mean()).astype(np.uint8)
    except Exception:
        return None


def deduplicar(imagenes, umbral=10):
    """
    Filtra imagenes casi-identicas (Hamming distance < umbral sobre 256 bits).
    No modifica archivos en disco; solo retorna la lista filtrada.
    """
    hashes, paths = [], []
    for img in imagenes:
        h = _ahash(img)
        if h is not None:
            hashes.append(h)
            paths.append(img)
    if not hashes:
        return imagenes, 0
    H = np.stack(hashes)
    keep = np.ones(len(paths), dtype=bool)
    for i in range(len(paths)):
        if not keep[i]:
            continue
        dists = np.sum(H[i + 1:] != H[i], axis=1)
        keep[np.where(dists < umbral)[0] + (i + 1)] = False
    unicos = [paths[i] for i in range(len(paths)) if keep[i]]
    return unicos, len(paths) - len(unicos)


# ─── Crops offline ────────────────────────────────────────────────────────────

def generar_crops_offline(img_path, dir_destino, stem, sufijo, rng, n_crops=3):
    """Genera crops aleatorios (50-80% del original) para forzar aprendizaje de texturas."""
    img = Image.open(img_path)
    w, h = img.size
    guardados = 0
    for c in range(n_crops):
        frac = rng.uniform(0.5, 0.8)
        cw, ch = int(w * frac), int(h * frac)
        x = rng.randint(0, w - cw)
        y = rng.randint(0, h - ch)
        crop = img.crop((x, y, x + cw, y + ch))
        (dir_destino / f"{stem}_crop{c}{sufijo}").parent.mkdir(parents=True, exist_ok=True)
        crop.save(dir_destino / f"{stem}_crop{c}{sufijo}", quality=95)
        guardados += 1
    return guardados


# ─── Pipeline principal ───────────────────────────────────────────────────────

def preparar_datos(
    ruta_origen="raw",
    ruta_destino="dataset_cacao",
    val_por_clase=50,
    test_por_clase=15,
    dedup_umbral=10,
    max_ratio=2.5,
):
    dir_origen = Path(ruta_origen)
    dir_destino = Path(ruta_destino)

    if not dir_origen.exists():
        print(f"Error: No se encontro '{dir_origen}'.")
        return

    clases = sorted([d.name for d in dir_origen.iterdir() if d.is_dir()])
    if not clases:
        print("Error: No se encontraron subcarpetas en 'raw/'.")
        return

    extensiones_validas = {".jpg", ".jpeg", ".png", ".webp"}
    rng = random.Random(42)

    # ── Fase 1: Escanear + deduplicar ────────────────────────────────────────
    imagenes_por_clase = {}
    print("Escaneando y deduplicando...")
    for clase in clases:
        imgs = sorted([p for p in (dir_origen / clase).iterdir()
                        if p.suffix.lower() in extensiones_validas])
        n_orig = len(imgs)
        imgs, n_dup = deduplicar(imgs, umbral=dedup_umbral)
        rng.shuffle(imgs)
        imagenes_por_clase[clase] = imgs
        dup_str = f" ({n_dup} duplicados filtrados)" if n_dup else ""
        print(f"  {clase}: {n_orig} → {len(imgs)} unicas{dup_str}")

    # ── Fase 1b: Subsampling de clase mayoritaria si excede max_ratio ────────
    min_count = min(len(imgs) for imgs in imagenes_por_clase.values())
    max_allowed = int(min_count * max_ratio)
    for clase, imgs in imagenes_por_clase.items():
        if len(imgs) > max_allowed:
            rng.shuffle(imgs)
            removed = len(imgs) - max_allowed
            imagenes_por_clase[clase] = imgs[:max_allowed]
            print(f"  {clase}: subsampled {len(imgs)+removed} → {max_allowed} "
                  f"(cap {max_ratio}x clase minoritaria)")

    # ── Fase 2: Split balanceado test / val / train ──────────────────────────
    min_clase = min(len(imgs) for imgs in imagenes_por_clase.values())
    n_test = min(test_por_clase, max(5, min_clase // 10))
    n_val  = min(val_por_clase, max(10, (min_clase - n_test) // 4))

    print(f"\nSplit balanceado: {n_test} test + {n_val} val por clase")

    splits = {"train": {}, "val": {}, "test": {}}
    for clase, imgs in imagenes_por_clase.items():
        splits["test"][clase]  = imgs[:n_test]
        splits["val"][clase]   = imgs[n_test : n_test + n_val]
        splits["train"][clase] = imgs[n_test + n_val:]

    # ── Fase 3: Oversampling clase minoritaria en train ───────────────────────
    conteos_train = {c: len(imgs) for c, imgs in splits["train"].items()}
    max_train = max(conteos_train.values())

    print("\nDistribucion (imagenes unicas):")
    for c in clases:
        print(f"  {c}: {conteos_train[c]} train / {len(splits['val'][c])} val / "
              f"{len(splits['test'][c])} test")

    oversampled = {}
    for clase, imgs in splits["train"].items():
        if len(imgs) < max_train:
            deficit = max_train - len(imgs)
            oversampled[clase] = imgs + [rng.choice(imgs) for _ in range(deficit)]
            print(f"\nOversampling '{clase}': {len(imgs)} → {len(oversampled[clase])} "
                  f"(+{deficit} duplicados)")
        else:
            oversampled[clase] = imgs

    # ── Fase 4: Copiar archivos + crops offline ──────────────────────────────
    if dir_destino.exists():
        shutil.rmtree(dir_destino)
        print(f"\nDirectorio '{dir_destino}' limpiado.")

    print("\nCopiando imagenes y generando crops...")
    for clase in clases:
        for split_name in ("train", "val", "test"):
            (dir_destino / split_name / clase).mkdir(parents=True, exist_ok=True)

        # Train
        dir_train = dir_destino / "train" / clase
        total_crops = 0
        contador = Counter()
        for img in oversampled[clase]:
            contador[img.name] += 1
            if contador[img.name] == 1:
                shutil.copy2(img, dir_train / img.name)
                total_crops += generar_crops_offline(
                    img, dir_train, img.stem, img.suffix, rng, n_crops=3)
            else:
                stem = f"{img.stem}_dup{contador[img.name] - 1}"
                shutil.copy2(img, dir_train / f"{stem}{img.suffix}")

        # Val y Test (limpios)
        for img in splits["val"][clase]:
            shutil.copy2(img, dir_destino / "val" / clase / img.name)
        for img in splits["test"][clase]:
            shutil.copy2(img, dir_destino / "test" / clase / img.name)

        nt = len(list((dir_destino / "train" / clase).iterdir()))
        nv = len(list((dir_destino / "val" / clase).iterdir()))
        ns = len(list((dir_destino / "test" / clase).iterdir()))
        print(f"  {clase}: {nt} train ({total_crops} crops) / {nv} val / {ns} test")

    # ── Resumen ──────────────────────────────────────────────────────────────
    t = sum(len(list((dir_destino / "train" / c).iterdir())) for c in clases)
    v = sum(len(list((dir_destino / "val"   / c).iterdir())) for c in clases)
    s = sum(len(list((dir_destino / "test"  / c).iterdir())) for c in clases)
    print(f"\nTotal: {t} train + {v} val + {s} test = {t + v + s}")
    print("Preprocesamiento finalizado.")


if __name__ == "__main__":
    preparar_datos()
