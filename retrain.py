import os
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 5

def consolidar_correcciones(carpeta_origen, carpeta_destino_base):
    """
    Mueve automáticamente las imágenes de corrección a las carpetas del dataset original
    para que las dimensiones de clases se mantengan idénticas.
    """
    if not os.path.exists(carpeta_origen):
        return False
    
    hay_archivos = False
    # Recorrer subcarpetas de correcciones (ej: '18-20', 'happy')
    for subcarpeta in os.listdir(carpeta_origen):
        ruta_sub_origen = os.path.join(carpeta_origen, subcarpeta)
        
        if os.path.isdir(ruta_sub_origen):
            ruta_sub_destino = os.path.join(carpeta_destino_base, subcarpeta)
            os.makedirs(ruta_sub_destino, exist_ok=True)
            
            # Mover cada archivo de imagen
            for archivo in os.listdir(ruta_sub_origen):
                origen_archivo = os.path.join(ruta_sub_origen, archivo)
                if os.path.isfile(origen_archivo):
                    # Evitar colisión de nombres agregando un prefijo fix_
                    nuevo_nombre = f"fix_{archivo}"
                    destino_archivo = os.path.join(ruta_sub_destino, nuevo_nombre)
                    
                    shutil.move(origen_archivo, destino_archivo)
                    hay_archivos = True
                    
    return hay_archivos

# ==========================================
# 1. CONSOLIDAR Y REENTRENAR EDAD
# ==========================================
print("\n--- Procesando Correcciones de EDAD ---")
hubo_cambios_edad = consolidar_correcciones("correcciones/edad", "datasets/edades")

if hubo_cambios_edad:
    print("🔄 Nuevas imágenes de edad integradas al dataset. Reentrenando...")
    
    generador_edad = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=10,
        zoom_range=0.1,
        horizontal_flip=True
    )
    
    edad_train = generador_edad.flow_from_directory(
        "datasets/edades",
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training"
    )
    
    edad_val = generador_edad.flow_from_directory(
        "datasets/edades",
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation"
    )
    
    if os.path.exists("modelo/edad.keras"):
        modelo_edad = load_model("modelo/edad.keras")
        modelo_edad.fit(edad_train, validation_data=edad_val, epochs=EPOCHS)
        modelo_edad.save("modelo/edad.keras")
        print("✅ Modelo de EDAD actualizado y guardado con éxito.")
    else:
        print("❌ Error: No se encontró 'modelo/edad.keras' base para reentrenar.")
else:
    print("ℹ️ No se detectaron nuevas correcciones pendientes para EDAD.")


# ==========================================
# 2. CONSOLIDAR Y REENTRENAR EMOCIÓN
# ==========================================
print("\n--- Procesando Correcciones de EMOCIONES ---")
hubo_cambios_emo = consolidar_correcciones("correcciones/emocion", "datasets/emociones_procesado")

if hubo_cambios_emo:
    print("🔄 Nuevas imágenes de emociones integradas al dataset. Reentrenando...")
    
    generador_emo = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=10,
        zoom_range=0.1,
        horizontal_flip=True
    )
    
    emo_train = generador_emo.flow_from_directory(
        "datasets/emociones_procesado",
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training"
    )
    
    emo_val = generador_emo.flow_from_directory(
        "datasets/emociones_procesado",
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation"
    )
    
    if os.path.exists("modelo/emocion.keras"):
        modelo_emo = load_model("modelo/emocion.keras")
        modelo_emo.fit(emo_train, validation_data=emo_val, epochs=EPOCHS)
        modelo_emo.save("modelo/emocion.keras")
        print("✅ Modelo de EMOCIONES actualizado y guardado con éxito.")
    else:
        print("❌ Error: No se encontró 'modelo/emocion.keras' base para reentrenar.")
else:
    print("ℹ️ No se detectaron nuevas correcciones pendientes para EMOCIONES.")

print("\n🚀 Proceso de reentrenamiento completado.")
