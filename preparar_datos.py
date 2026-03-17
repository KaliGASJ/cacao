import os
import shutil
import random
from pathlib import Path

def preparar_datos(ruta_origen, ruta_destino, proporcion_train=0.8):
    """
    Toma imágenes de una carpeta raw/ y las distribuye aleatoriamente en
    las carpetas train/ y val/ para entrenamiento de YOLO clasificadores.
    
    Args:
        ruta_origen (str): Ruta donde están las imágenes originales, separadas por clase (ej: raw/sano, raw/moniliasis).
        ruta_destino (str): Ruta principal del dataset estructurado para YOLO (ej: dataset_cacao).
        proporcion_train (float): Porcentaje de imágenes que irán a entrenamiento (por defecto 80%).
    """
    # Definimos rutas usando pathlib para mayor compatibilidad y limpieza
    dir_origen = Path(ruta_origen)
    dir_destino = Path(ruta_destino)
    
    # Validamos que el directorio de origen exista
    if not dir_origen.exists():
        print(f"Error: La ruta de origen '{dir_origen}' no existe.")
        print("Por favor, crea la carpeta 'raw' y coloca dentro subcarpetas para cada clase (ej: 'raw/sano', 'raw/moniliasis').")
        return

    # Listamos todas las subcarpetas dentro de la ruta origen (las clases: sano, moniliasis)
    clases = [d.name for d in dir_origen.iterdir() if d.is_dir()]
    
    if not clases:
        print(f"Error: No se encontraron subcarpetas (clases) en '{dir_origen}'.")
        return

    print(f"Clases detectadas: {clases}")
    print("Iniciando procesamiento de imágenes...")

    total_movidas = 0

    for clase in clases:
        # Rutas de las imágenes de esta clase en raw/
        ruta_clase_origen = dir_origen / clase
        
        # Filtramos solo archivos que parezcan imágenes
        extensiones_validas = {'.jpg', '.jpeg', '.png', '.webp'}
        imagenes = [img for img in ruta_clase_origen.iterdir() if img.suffix.lower() in extensiones_validas]
        
        # Mezclamos aleatoriamente la lista de imágenes para garantizar variedad
        random.seed(42)  # Semilla fija para asegurar reproducibilidad si se corre varias veces
        random.shuffle(imagenes)
        
        # Calculamos la cantidad para train (el resto será para val)
        cantidad_train = int(len(imagenes) * proporcion_train)
        
        # Separamos las listas
        imagenes_train = imagenes[:cantidad_train]
        imagenes_val = imagenes[cantidad_train:]
        
        # Rutas destino para esta clase en específico
        rutas_destino_clase = {
            'train': dir_destino / 'train' / clase,
            'val': dir_destino / 'val' / clase
        }
        
        # Nos aseguramos que los directorios destino existan (aunque ya los creaste en bash)
        for particion, ruta in rutas_destino_clase.items():
            ruta.mkdir(parents=True, exist_ok=True)
            
        # Función auxiliar para copiar/mover archivos
        def procesar_lote(lote, destino, nombre_particion):
            nonlocal total_movidas
            for img in lote:
                destino_final = destino / img.name
                # Movemos el archivo en lugar de copiar para ahorrar espacio. 
                # (Si prefieres conservar el raw y solo copiar, cambia 'shutil.move' por 'shutil.copy2')
                shutil.copy2(img, destino_final) 
                total_movidas += 1
            print(f"  - {clase} -> {nombre_particion}: {len(lote)} imágenes procesadas.")

        # Ejecutamos el traslado
        procesar_lote(imagenes_train, rutas_destino_clase['train'], 'train')
        procesar_lote(imagenes_val, rutas_destino_clase['val'], 'val')

    print(f"\n¡Proceso finalizado! Se distribuyeron un total de {total_movidas} imágenes exitosamente.")
    print(f"Estructura lista en '{dir_destino}' para entrenar con YOLO.")

if __name__ == '__main__':
    # Ejecutamos la función asumiendo que el usuario creó una carpeta 'raw'
    # con subcarpetas 'sano' y 'moniliasis'
    preparar_datos(ruta_origen="raw", ruta_destino="dataset_cacao", proporcion_train=0.8)
