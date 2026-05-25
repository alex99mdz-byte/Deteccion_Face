from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import os

IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 5

# =========================
# REENTRENAR EDAD
# =========================

print("\nReentrenando EDAD...")

generador_edad = ImageDataGenerator(

    rescale=1./255,

    validation_split=0.2,

    rotation_range=10,

    zoom_range=0.1,

    horizontal_flip=True

)

edad_train = generador_edad.flow_from_directory(

    "datasets/edades",

    target_size=(IMG_SIZE,IMG_SIZE),

    batch_size=BATCH_SIZE,

    class_mode="categorical",

    subset="training"

)

edad_val = generador_edad.flow_from_directory(

    "datasets/edades",

    target_size=(IMG_SIZE,IMG_SIZE),

    batch_size=BATCH_SIZE,

    class_mode="categorical",

    subset="validation"

)

# Agregar correcciones
if os.path.exists("correcciones/edad"):

    edad_extra = generador_edad.flow_from_directory(

        "correcciones/edad",

        target_size=(IMG_SIZE,IMG_SIZE),

        batch_size=BATCH_SIZE,

        class_mode="categorical",

        shuffle=True

    )

else:

    edad_extra = None

modelo_edad = load_model(

    "modelo/edad.keras"

)

if edad_extra:

    modelo_edad.fit(

        edad_extra,

        epochs=EPOCHS

    )

modelo_edad.fit(

    edad_train,

    validation_data=edad_val,

    epochs=EPOCHS

)

modelo_edad.save(

    "modelo/edad.keras"

)

print(

"Edad actualizada"

)

# =========================
# REENTRENAR EMOCION
# =========================

print(

"\nReentrenando EMOCIONES..."

)

generador_emo = ImageDataGenerator(

    rescale=1./255,

    validation_split=0.2,

    rotation_range=10,

    zoom_range=0.1,

    horizontal_flip=True

)

emo_train = generador_emo.flow_from_directory(

    "datasets/emociones_procesado",

    target_size=(IMG_SIZE,IMG_SIZE),

    batch_size=BATCH_SIZE,

    class_mode="categorical",

    subset="training"

)

emo_val = generador_emo.flow_from_directory(

    "datasets/emociones_procesado",

    target_size=(IMG_SIZE,IMG_SIZE),

    batch_size=BATCH_SIZE,

    class_mode="categorical",

    subset="validation"

)

if os.path.exists(

    "correcciones/emocion"

):

    emo_extra = generador_emo.flow_from_directory(

        "correcciones/emocion",

        target_size=(IMG_SIZE,IMG_SIZE),

        batch_size=BATCH_SIZE,

        class_mode="categorical",

        shuffle=True

    )

else:

    emo_extra=None

modelo_emo = load_model(

    "modelo/emocion.keras"

)

if emo_extra:

    modelo_emo.fit(

        emo_extra,

        epochs=EPOCHS

    )

modelo_emo.fit(

    emo_train,

    validation_data=emo_val,

    epochs=EPOCHS

)

modelo_emo.save(

    "modelo/emocion.keras"

)

print(

"Emociones actualizadas"

)

print(

"\nIA reentrenada correctamente"
)
