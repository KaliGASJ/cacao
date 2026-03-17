import shutil
import random
from pathlib import Path

def preparar_datos(ruta_origen="raw", ruta_destino="dataset_cacao", proporcion_train=0.8):
    """
    Distribuye las imágenes de raw a train y val.
    """
    dir_origen = Path(ruta_origen)
    dir_destino = Path(ruta_destino)
    
    # Validar origen
    if not dir_origen.exists():
        print(f"Error: No se encontró '{dir_origen}'.")
        return

    clases = [d.name for d in dir_origen.iterdir() if d.is_dir()]
    if not clases:
        return

    print("Procesando imágenes...")
    
    for clase in clases:
        # Filtrar imágenes
        ruta_clase_origen = dir_origen / clase
        imagenes = [img for img in ruta_clase_origen.iterdir() if img.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'}]
        
        # Mezclar para aleatoriedad
        random.seed(42)
        random.shuffle(imagenes)
        
        # Calcular partición
        cant_train = int(len(imagenes) * proporcion_train)
        imagenes_train = imagenes[:cant_train]
        imagenes_val = imagenes[cant_train:]
        
        # Generar directorios destino
        rutas_destino = {
            'train': dir_destino / 'train' / clase,
            'val': dir_destino / 'val' / clase
        }
        for ruta in rutas_destino.values():
            ruta.mkdir(parents=True, exist_ok=True)
            
        # Copiar archivos
        for img in imagenes_train:
            shutil.copy2(img, rutas_destino['train'] / img.name)
        for img in imagenes_val:
            shutil.copy2(img, rutas_destino['val'] / img.name)
            
        print(f" - {clase}: {len(imagenes_train)} a train, {len(imagenes_val)} a val.")
        
    print("Preprocesamiento finalizado.")

if __name__ == '__main__':
    preparar_datos()
