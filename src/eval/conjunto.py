"""Lectura de un conjunto COCO con cajas de referencia de una sola clase (dataset de la etapa 1)."""

import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

NOMBRE_ANOTACIONES = "_annotations.coco.json"


@dataclass(frozen=True)
class ImagenGT:
    """Una imagen del conjunto con sus cajas de referencia.

    ``cajas`` es una matriz (N, 4) con ``(x1, y1, x2, y2)`` en píxeles (vacía si no hay lechugas).
    """

    nombre: str
    ruta: Path
    ancho: int
    alto: int
    cajas: np.ndarray


def cargar_conjunto_coco(carpeta: str | Path, clase: str) -> list[ImagenGT]:
    """Lee ``<carpeta>/_annotations.coco.json`` y devuelve las imágenes con sus cajas.

    En lenguaje llano: el formato COCO guarda cada caja como ``[x, y, ancho, alto]``; aquí se
    convierte a esquinas ``(x1, y1, x2, y2)``, que es lo que usan el IoU y el emparejamiento. Solo
    cuentan las anotaciones de la categoría ``clase``; las marcadas ``iscrowd`` (un grupo de
    objetos, no uno) se ignoran con un aviso. Las imágenes salen ordenadas por su ``id``.

    Errores: ``FileNotFoundError`` si falta el JSON o alguna imagen; ``ValueError`` si el JSON no
    tiene la forma esperada o no tiene la categoría ``clase``.
    """
    carpeta = Path(carpeta)
    ruta_json = carpeta / NOMBRE_ANOTACIONES
    if not ruta_json.is_file():
        raise FileNotFoundError(f"No existe {NOMBRE_ANOTACIONES} en la partición {carpeta.name}")
    datos = json.loads(ruta_json.read_text(encoding="utf-8"))
    for seccion in ("images", "annotations", "categories"):
        if not isinstance(datos, dict) or not isinstance(datos.get(seccion), list):
            raise ValueError(f"{NOMBRE_ANOTACIONES}: falta la sección '{seccion}'")

    ids = [c["id"] for c in datos["categories"] if c.get("name") == clase]
    if not ids:
        existentes = ", ".join(str(c.get("name")) for c in datos["categories"])
        raise ValueError(f"La categoría {clase!r} no está en el conjunto (hay: {existentes})")
    id_clase = ids[0]

    cajas_por_imagen: dict[int, list[list[float]]] = {}
    con_crowd = 0
    for anotacion in datos["annotations"]:
        if anotacion["category_id"] != id_clase:
            continue
        if anotacion.get("iscrowd", 0):
            con_crowd += 1
            continue
        x, y, ancho, alto = anotacion["bbox"]
        cajas_por_imagen.setdefault(anotacion["image_id"], []).append([x, y, x + ancho, y + alto])
    if con_crowd:
        warnings.warn(
            f"Se ignoran {con_crowd} anotaciones con iscrowd distinto de 0",
            UserWarning,
            stacklevel=2,
        )

    imagenes = []
    for imagen in sorted(datos["images"], key=lambda i: i["id"]):
        ruta = carpeta / imagen["file_name"]
        if not ruta.is_file():
            raise FileNotFoundError(f"Falta la imagen {imagen['file_name']} en {carpeta.name}")
        cajas = np.array(cajas_por_imagen.get(imagen["id"], []), dtype=float).reshape(-1, 4)
        imagenes.append(
            ImagenGT(
                nombre=imagen["file_name"],
                ruta=ruta,
                ancho=int(imagen["width"]),
                alto=int(imagen["height"]),
                cajas=cajas,
            )
        )
    return imagenes
