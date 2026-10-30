"""Formato COCO: lectura de ZIP exportados de Roboflow y geometría de polígonos.

Este módulo solo sabe de COCO (sin lógica del proyecto): leer los ``_annotations.coco.json``
de un ZIP, contar clases, calcular cajas y áreas desde polígonos y escribir un JSON COCO.
Un ZIP de Roboflow tiene carpetas ``train``, ``valid`` y ``test``, cada una con sus imágenes y
un único ``_annotations.coco.json``.
"""

import json
import zipfile
from collections import Counter
from pathlib import Path

# Carpetas que puede tener un ZIP de Roboflow, en el orden en que se recorren siempre.
CARPETAS = ("train", "valid", "test")

NOMBRE_JSON = "_annotations.coco.json"


def leer_anotaciones_zip(zf: zipfile.ZipFile) -> dict[str, dict]:
    """Lee el JSON COCO de cada carpeta del ZIP y devuelve ``{carpeta: contenido}``.

    En lenguaje llano: busca ``train/_annotations.coco.json``, ``valid/...`` y ``test/...``
    dentro del ZIP (sin extraer nada) y los carga. Solo salen las carpetas que existen, en el
    orden de ``CARPETAS``. ``ValueError`` si el ZIP no tiene ninguno.
    """
    nombres = set(zf.namelist())
    salida: dict[str, dict] = {}
    for carpeta in CARPETAS:
        ruta = f"{carpeta}/{NOMBRE_JSON}"
        if ruta in nombres:
            salida[carpeta] = json.loads(zf.read(ruta).decode("utf-8"))
    if not salida:
        raise ValueError(
            f"El ZIP no contiene ningún {NOMBRE_JSON} en {', '.join(CARPETAS)}: no parece "
            "una exportación COCO de Roboflow"
        )
    return salida


def nombres_de_categorias(coco: dict) -> dict[int, str]:
    """Devuelve ``{id de categoría: nombre}`` de un JSON COCO."""
    return {categoria["id"]: categoria["name"] for categoria in coco.get("categories", [])}


def clases_con_anotaciones(coco: dict) -> Counter:
    """Cuenta las anotaciones de cada clase, identificada por su NOMBRE.

    En lenguaje llano: Roboflow mete en ``categories`` una categoría "padre" sin anotaciones
    (y a veces con el mismo nombre que la real, con otro id). Contando anotaciones por nombre,
    esas categorías vacías no aparecen y las repetidas se suman.
    """
    nombres = nombres_de_categorias(coco)
    return Counter(nombres[anotacion["category_id"]] for anotacion in coco["annotations"])


def poligono_valido(poligono: list[float]) -> bool:
    """True si el polígono tiene al menos 3 puntos y un área distinta de cero."""
    if len(poligono) < 6 or len(poligono) % 2:
        return False
    return _area_poligono(poligono) > 0


def _area_poligono(poligono: list[float]) -> float:
    """Área de un polígono con la fórmula del cordón (shoelace); siempre positiva."""
    xs, ys = poligono[0::2], poligono[1::2]
    suma = 0.0
    for i in range(len(xs)):
        j = (i + 1) % len(xs)
        suma += xs[i] * ys[j] - xs[j] * ys[i]
    return abs(suma) / 2


def area_poligonos(poligonos: list[list[float]]) -> float:
    """Suma de las áreas de los polígonos de una anotación."""
    return sum(_area_poligono(poligono) for poligono in poligonos)


def bbox_desde_poligonos(poligonos: list[list[float]]) -> tuple[float, float, float, float]:
    """Caja ``(x, y, ancho, alto)`` que abarca todos los puntos de los polígonos.

    En lenguaje llano: se toman las x e y más pequeñas y más grandes de todos los vértices; la
    caja va de la esquina (x mínima, y mínima) y su ancho y alto son la diferencia con los máximos.
    """
    xs = [x for poligono in poligonos for x in poligono[0::2]]
    ys = [y for poligono in poligonos for y in poligono[1::2]]
    return (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))


def escribir_coco(
    ruta: str | Path, imagenes: list[dict], anotaciones: list[dict], categorias: list[dict]
) -> None:
    """Escribe un JSON COCO (utf-8) con imágenes, anotaciones y categorías."""
    contenido = {"images": imagenes, "annotations": anotaciones, "categories": categorias}
    Path(ruta).write_text(json.dumps(contenido, ensure_ascii=False, indent=1), encoding="utf-8")
