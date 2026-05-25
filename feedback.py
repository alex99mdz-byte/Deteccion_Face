import os
import shutil

CORRECCIONES="correcciones"

def guardar_correccion(

    ruta_imagen,

    edad=None,

    emocion=None

):

    if edad:

        carpeta = os.path.join(

            CORRECCIONES,

            "edad",

            edad

        )

        os.makedirs(

            carpeta,

            exist_ok=True

        )

        shutil.copy2(
            ruta_imagen,carpeta
        )

    if emocion:
        carpeta = os.path.join(
            CORRECCIONES,"emocion",emocion
        )

        os.makedirs(
            carpeta,exist_ok=True
        )

        shutil.copy2(
            ruta_imagen,carpeta
        )

print(
"Feedback listo"
)
