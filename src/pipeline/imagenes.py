"""Lectura y escritura de imágenes que funciona con tildes y eñes en la ruta (Windows).

``cv2.imread`` y ``cv2.imwrite`` fallan en Windows si la ruta tiene caracteres no ASCII.
Por eso aquí se leen los bytes con numpy y se decodifican/codifican en memoria.
"""

from pathlib import Path

import cv2
import numpy as np

# Extensiones de imagen que acepta el pipeline (se comparan en minúsculas).
EXTENSIONES = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp"})


def leer_imagen(ruta: str | Path) -> np.ndarray:
    """Lee una imagen y la devuelve como array BGR de tres canales (alto, ancho, 3).

    En lenguaje llano: carga el archivo entero en memoria y se lo da a OpenCV para que lo
    interprete; así la ruta no pasa por OpenCV y no importan las tildes ni las eñes.
    Una imagen en gris se convierte a tres canales.

    Errores: ``OSError`` si el archivo no existe o no se puede abrir; ``ValueError`` si el
    archivo está vacío o no es una imagen válida.
    """
    ruta = Path(ruta)
    datos = np.fromfile(ruta, dtype=np.uint8)
    if datos.size == 0:
        raise ValueError(f"el archivo está vacío: {ruta.name}")
    try:
        imagen = cv2.imdecode(datos, cv2.IMREAD_COLOR)
    except cv2.error as error:
        raise ValueError(f"no se puede decodificar {ruta.name}: {error}") from error
    if imagen is None:
        raise ValueError(f"no es una imagen válida: {ruta.name}")
    return imagen


def guardar_imagen(ruta: str | Path, imagen: np.ndarray) -> None:
    """Guarda una imagen; el formato lo decide la extensión de la ruta (.jpg, .png...).

    En lenguaje llano: OpenCV convierte la imagen a los bytes del formato elegido en
    memoria y esos bytes se escriben en el archivo con numpy, sin pasar la ruta a OpenCV.

    Error: ``ValueError`` si la extensión no corresponde a ningún formato conocido.
    """
    ruta = Path(ruta)
    try:
        correcto, datos = cv2.imencode(ruta.suffix, imagen)
    except cv2.error as error:
        raise ValueError(f"no se puede codificar con la extensión '{ruta.suffix}'") from error
    if not correcto:
        raise ValueError(f"no se pudo codificar la imagen como '{ruta.suffix}'")
    datos.tofile(ruta)
