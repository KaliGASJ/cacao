"""
preparar_datos.py
─────────────────
Distribuye automáticamente las imágenes de la carpeta 'raw/' en
subcarpetas 'train/' y 'val/' dentro de 'dataset_cacao/', respetando
la estructura de directorios que requiere Ultralytics YOLO para
clasificación de imágenes.

Uso:
    python3 preparar_datos.py
"""

import shutil
import random
from pathlib import Path


def preparar_datos(ruta_origen="raw", ruta_destino="dataset_cacao", proporcion_train=0.8):
    """
    Lee cada subcarpeta de 'ruta_origen' como una clase, mezcla sus
    imágenes aleatoriamente y las copia a 'ruta_destino' divididas
    en conjuntos de entrenamiento (80 %) y validación (20 %).

    Args:
        ruta_origen:      Carpeta raíz con subcarpetas por clase (ej. raw/sano, raw/moniliasis).
        ruta_destino:     Carpeta donde se generará la estructura train/val.
        proporcion_train: Fracción de imágenes destinadas a entrenamiento (0.0 – 1.0).
    """
    dir_origen = Path(ruta_origen)
    dir_destino = Path(ruta_destino)

    # Validar que la carpeta de origen exista
    if not dir_origen.exists():
        print(f"Error: No se encontró la carpeta '{dir_origen}'.")
        return

    # Detectar clases (cada subcarpeta = una clase)
    clases = [d.name for d in dir_origen.iterdir() if d.is_dir()]
    if not clases:
        print("Error: No se encontraron subcarpetas de clases dentro de 'raw/'.")
        return

    print("Procesando imágenes...")

    for clase in clases:
        # Listar solo archivos de imagen válidos
        ruta_clase_origen = dir_origen / clase
        extensiones_validas = {".jpg", ".jpeg", ".png", ".webp"}
        imagenes = [
            img for img in ruta_clase_origen.iterdir()
            if img.suffix.lower() in extensiones_validas
        ]

        # Mezclar con semilla fija para reproducibilidad
        random.seed(42)
        random.shuffle(imagenes)

        # Dividir en entrenamiento y validación
        cant_train = int(len(imagenes) * proporcion_train)
        imagenes_train = imagenes[:cant_train]
        imagenes_val = imagenes[cant_train:]

        # Crear directorios de destino si no existen
        rutas_destino = {
            "train": dir_destino / "train" / clase,
            "val": dir_destino / "val" / clase,
        }
        for ruta in rutas_destino.values():
            ruta.mkdir(parents=True, exist_ok=True)

        # Copiar imágenes preservando metadatos
        for img in imagenes_train:
            shutil.copy2(img, rutas_destino["train"] / img.name)
        for img in imagenes_val:
            shutil.copy2(img, rutas_destino["val"] / img.name)

        print(f" - {clase}: {len(imagenes_train)} a train, {len(imagenes_val)} a val.")

    print("Preprocesamiento finalizado.")


if __name__ == "__main__":
    preparar_datos()
