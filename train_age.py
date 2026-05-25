from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout

IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 10

# Generador de imágenes
generador = ImageDataGenerator(

    rescale=1./255,

    validation_split=0.2,

    rotation_range=10,

    zoom_range=0.1,

    horizontal_flip=True

)

# Dataset entrenamiento
train = generador.flow_from_directory(

    "datasets/edades",

    target_size=(IMG_SIZE, IMG_SIZE),

    batch_size=BATCH_SIZE,

    class_mode="categorical",

    subset="training"

)

# Dataset validación
validacion = generador.flow_from_directory(

    "datasets/edades",

    target_size=(IMG_SIZE, IMG_SIZE),

    batch_size=BATCH_SIZE,

    class_mode="categorical",

    subset="validation"

)

print("\nCategorias detectadas:")
print(train.class_indices)

# Detectar automáticamente cuántas clases existen
NUM_CLASES = len(train.class_indices)

print(f"\nNumero clases: {NUM_CLASES}")

# Crear modelo
modelo = Sequential([

    Input(shape=(IMG_SIZE, IMG_SIZE, 3)),

    Conv2D(
        32,
        (3,3),
        activation="relu"
    ),

    MaxPooling2D(2,2),

    Conv2D(
        64,
        (3,3),
        activation="relu"
    ),

    MaxPooling2D(2,2),

    Conv2D(
        128,
        (3,3),
        activation="relu"
    ),

    MaxPooling2D(2,2),

    Flatten(),

    Dense(
        128,
        activation="relu"
    ),

    Dropout(0.5),

    Dense(
        NUM_CLASES,
        activation="softmax"
    )

])

# Compilar modelo
modelo.compile(

    optimizer="adam",

    loss="categorical_crossentropy",

    metrics=["accuracy"]

)

modelo.summary()

print("\nEntrenando modelo...\n")

# Entrenamiento
historial = modelo.fit(

    train,

    validation_data=validacion,

    epochs=EPOCHS

)

# Guardar modelo
modelo.save(

    "modelo/edad.keras"

)

print("\nModelo guardado correctamente")
