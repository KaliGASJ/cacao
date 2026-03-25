"""
preparar_datos.py
-----------------
Distribuye las imagenes de 'raw/' en 'dataset_cacao/{train,val}/'
con split estratificado 80/20, oversampling de la clase minoritaria,
y generacion de crops aleatorios para romper el sesgo de composicion.

Uso:
    python3 preparar_datos.py
"""

import shutil
import random
from pathlib import Path
from collections import Counter
from PIL import Image


def generar_crops_offline(img_path, dir_destino, stem, sufijo, rng, n_crops=3):
    """
    Genera crops aleatorios de una imagen y los guarda en disco.
    Esto fuerza al modelo a aprender de regiones parciales (textura)
    en vez del contexto global (mano, fondo, composicion).
    """
    img = Image.open(img_path)
    w, h = img.size
    guardados = 0

    for c in range(n_crops):
        # Crop entre 50% y 80% del tamano original, posicion aleatoria
        frac = rng.uniform(0.5, 0.8)
        cw, ch = int(w * frac), int(h * frac)
        x = rng.randint(0, w - cw)
        y = rng.randint(0, h - ch)
        crop = img.crop((x, y, x + cw, y + ch))
        destino = dir_destino / f"{stem}_crop{c}{sufijo}"
        crop.save(destino, quality=95)
        guardados += 1

    return guardados


def preparar_datos(ruta_origen="raw", ruta_destino="dataset_cacao", val_por_clase=50):
    """
    val_por_clase: numero ABSOLUTO de imagenes reservadas para validacion por clase.
    Esto garantiza un val set balanceado independientemente del tamano de cada clase.
    El resto va a train (+ oversampling + crops).
    """
    dir_origen = Path(ruta_origen)
    dir_destino = Path(ruta_destino)

    if not dir_origen.exists():
        print(f"Error: No se encontro la carpeta '{dir_origen}'.")
        return

    clases = sorted([d.name for d in dir_origen.iterdir() if d.is_dir()])
    if not clases:
        print("Error: No se encontraron subcarpetas de clases dentro de 'raw/'.")
        return

    extensiones_validas = {".jpg", ".jpeg", ".png", ".webp"}
    rng = random.Random(42)

    conteo_por_clase = {}
    imagenes_train_por_clase = {}
    imagenes_val_por_clase = {}

    # --- Fase 1: Calcular tamaño de val balanceado entre clases ---
    imagenes_por_clase = {}
    for clase in clases:
        ruta_clase = dir_origen / clase
        imagenes = sorted([
            img for img in ruta_clase.iterdir()
            if img.suffix.lower() in extensiones_validas
        ])
        rng.shuffle(imagenes)
        imagenes_por_clase[clase] = imagenes
        conteo_por_clase[clase] = len(imagenes)

    # El val por clase = mínimo entre val_por_clase y el 20% de la clase más pequeña
    # Esto garantiza val perfectamente balanceado (misma N en todas las clases)
    n_val_global = min(
        val_por_clase,
        min(max(1, len(imgs) // 5) for imgs in imagenes_por_clase.values())
    )

    for clase, imagenes in imagenes_por_clase.items():
        imagenes_val_por_clase[clase] = imagenes[:n_val_global]
        imagenes_train_por_clase[clase] = imagenes[n_val_global:]

    # --- Fase 2: Oversampling de clase minoritaria en train ---
    conteos_train = {c: len(imgs) for c, imgs in imagenes_train_por_clase.items()}
    max_train = max(conteos_train.values())

    print("Distribucion original del dataset:")
    for clase in clases:
        total = conteo_por_clase[clase]
        n_train = conteos_train[clase]
        n_val = len(imagenes_val_por_clase[clase])
        print(f"  {clase}: {total} total ({n_train} train / {n_val} val)")
    print(f"  Val balanceado: {' + '.join(str(len(imagenes_val_por_clase[c]))+' '+c for c in clases)}")

    oversampled = {}
    for clase, imgs in imagenes_train_por_clase.items():
        if len(imgs) < max_train:
            deficit = max_train - len(imgs)
            extras = [rng.choice(imgs) for _ in range(deficit)]
            oversampled[clase] = imgs + extras
            print(f"\nOversampling '{clase}': {len(imgs)} -> {len(oversampled[clase])} "
                  f"(+{deficit} duplicados)")
        else:
            oversampled[clase] = imgs

    # --- Fase 3: Copiar archivos + generar crops ---
    if dir_destino.exists():
        shutil.rmtree(dir_destino)
        print(f"\nDirectorio '{dir_destino}' limpiado.")

    print("\nCopiando imagenes y generando crops...")

    for clase in clases:
        dir_train = dir_destino / "train" / clase
        dir_val = dir_destino / "val" / clase
        dir_train.mkdir(parents=True, exist_ok=True)
        dir_val.mkdir(parents=True, exist_ok=True)

        total_crops = 0

        # Train (con oversampling + crops)
        contador_duplicados = Counter()
        for img in oversampled[clase]:
            contador_duplicados[img.name] += 1
            if contador_duplicados[img.name] == 1:
                destino = dir_train / img.name
                stem = img.stem
            else:
                stem = f"{img.stem}_dup{contador_duplicados[img.name] - 1}"
                destino = dir_train / f"{stem}{img.suffix}"
            shutil.copy2(img, destino)

            # Generar 3 crops aleatorios por imagen original (no duplicados)
            if contador_duplicados[img.name] == 1:
                total_crops += generar_crops_offline(
                    img, dir_train, img.stem, img.suffix, rng, n_crops=3
                )

        # Val (sin oversampling ni crops para evaluacion limpia)
        for img in imagenes_val_por_clase[clase]:
            shutil.copy2(img, dir_val / img.name)

        n_train_final = len(list(dir_train.iterdir()))
        n_val_final = len(list(dir_val.iterdir()))
        print(f"  {clase}: {n_train_final} train ({total_crops} crops) / {n_val_final} val")

    # --- Resumen final ---
    total_train = sum(len(list((dir_destino / "train" / c).iterdir())) for c in clases)
    total_val = sum(len(list((dir_destino / "val" / c).iterdir())) for c in clases)
    print(f"\nTotal: {total_train} train + {total_val} val = {total_train + total_val}")
    print("Preprocesamiento finalizado.")


if __name__ == "__main__":
    preparar_datos()
