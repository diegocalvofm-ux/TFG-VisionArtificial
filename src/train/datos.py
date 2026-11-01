"""Copia de trabajo del dataset de la etapa 1: subconjunto para el smoke y repeticiones de svodk.

El dataset exportado por el importador (``data/processed/etapa1_v1``) nunca se modifica. Para
cada entrenamiento se crea una COPIA DE TRABAJO en otra carpeta (fuera de ``runs/``): ahí se
aplican, solo a ``train``, el subconjunto del modo ``--smoke`` y las repeticiones de svodk.
``valid`` y ``test`` se copian siempre tal cual.
"""

import copy
import json
import random
import shutil
from pathlib import Path

CARPETAS = ("train", "valid", "test")
NOMBRE_JSON = "_annotations.coco.json"


def _leer_coco(carpeta: Path) -> dict:
    ruta = carpeta / NOMBRE_JSON
    if not ruta.is_file():
        raise FileNotFoundError(
            f"No existe {carpeta.name}/{NOMBRE_JSON} en el dataset: ¿se ejecutó el importador?"
        )
    return json.loads(ruta.read_text(encoding="utf-8"))


def subconjunto_smoke(coco: dict, n: int, semilla: int) -> dict:
    """Un COCO con solo ``n`` imágenes de ``coco`` (sorteadas con la semilla) y sus anotaciones.

    En lenguaje llano: se ordenan las imágenes por nombre, se sortean ``n`` con la semilla (así
    el resultado es siempre el mismo) y se queda cada una con todas sus anotaciones. Si ``n`` no
    es menor que el total, se devuelve todo. El COCO de entrada no se modifica.
    """
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError(f"smoke_imagenes debe ser un entero >= 1 (llegó {n!r})")
    ordenadas = sorted(coco["images"], key=lambda imagen: imagen["file_name"])
    if n < len(ordenadas):
        ordenadas = sorted(
            random.Random(semilla).sample(ordenadas, n), key=lambda imagen: imagen["file_name"]
        )
    ids = {imagen["id"] for imagen in ordenadas}
    salida = {clave: copy.deepcopy(valor) for clave, valor in coco.items()}
    salida["images"] = copy.deepcopy(ordenadas)
    salida["annotations"] = [copy.deepcopy(a) for a in coco["annotations"] if a["image_id"] in ids]
    return salida


def _repetir(coco: dict, prefijo: str, repeticiones: int) -> tuple[dict, list[tuple[str, str]]]:
    """Repite ``repeticiones`` veces en total las imágenes cuyo nombre empieza por ``prefijo``.

    Devuelve el COCO nuevo y la lista ``(nombre_nuevo, nombre_original)`` de los archivos que hay
    que copiar. Cada copia tiene un nombre de archivo propio (``svodk_0001__r2.jpg``), un id de
    imagen nuevo y las mismas anotaciones que la original, con ids de anotación nuevos.
    """
    elegidas = [i for i in coco["images"] if i["file_name"].startswith(prefijo)]
    if not elegidas:
        raise ValueError(
            f"repeticiones_svodk={repeticiones} pero no hay imágenes de train con el prefijo "
            f"'{prefijo}'"
        )
    anotaciones_de: dict[int, list[dict]] = {}
    for anotacion in coco["annotations"]:
        anotaciones_de.setdefault(anotacion["image_id"], []).append(anotacion)

    imagenes = list(coco["images"])
    anotaciones = list(coco["annotations"])
    siguiente_imagen = max((i["id"] for i in imagenes), default=0) + 1
    siguiente_anotacion = max((a["id"] for a in anotaciones), default=0) + 1
    copias: list[tuple[str, str]] = []
    for imagen in elegidas:
        original = Path(imagen["file_name"])
        for vez in range(2, repeticiones + 1):
            nombre = f"{original.stem}__r{vez}{original.suffix}"
            imagenes.append({**imagen, "id": siguiente_imagen, "file_name": nombre})
            for anotacion in anotaciones_de.get(imagen["id"], []):
                anotaciones.append(
                    {
                        **copy.deepcopy(anotacion),
                        "id": siguiente_anotacion,
                        "image_id": siguiente_imagen,
                    }
                )
                siguiente_anotacion += 1
            copias.append((nombre, imagen["file_name"]))
            siguiente_imagen += 1
    return {**coco, "images": imagenes, "annotations": anotaciones}, copias


def _contar(coco: dict, prefijo: str) -> dict:
    return {
        "imagenes": len(coco["images"]),
        "anotaciones": len(coco["annotations"]),
        "imagenes_con_prefijo": sum(i["file_name"].startswith(prefijo) for i in coco["images"]),
    }


def crear_copia_de_trabajo(
    dataset: str | Path,
    destino: str | Path,
    *,
    repeticiones_svodk: int,
    prefijo_svodk: str,
    smoke_imagenes: int | None,
    semilla: int,
) -> dict:
    """Crea la copia de trabajo del dataset y devuelve los recuentos de cada carpeta.

    En lenguaje llano: copia ``dataset`` (con carpetas train, valid y test) a ``destino``. En
    ``train`` primero se queda con ``smoke_imagenes`` imágenes si se pide (modo smoke) y luego cada
    imagen cuyo nombre empieza por ``prefijo_svodk`` aparece ``repeticiones_svodk`` veces en total
    (1 = sin repetir). ``valid`` y ``test`` se copian idénticos: nunca se repite nada ahí. Solo se
    copian los archivos que usa cada JSON. El dataset original no se toca.

    Devuelve ``{carpeta: {"imagenes", "anotaciones", "imagenes_con_prefijo"}}`` tal como queda la
    copia. Errores (``ValueError``): ``repeticiones_svodk`` no es un entero >= 1; hay repeticiones
    pero ninguna imagen de train lleva el prefijo; ``destino`` ya tiene contenido (no se borra
    nada: se usa otro ``exp-id`` o se borra a mano). ``FileNotFoundError`` si falta algún JSON.
    """
    if (
        isinstance(repeticiones_svodk, bool)
        or not isinstance(repeticiones_svodk, int)
        or repeticiones_svodk < 1
    ):
        raise ValueError(
            f"repeticiones_svodk debe ser un entero >= 1 (llegó {repeticiones_svodk!r})"
        )
    dataset, destino = Path(dataset), Path(destino)
    cocos = {carpeta: _leer_coco(dataset / carpeta) for carpeta in CARPETAS}
    if destino.exists() and any(destino.iterdir()):
        raise ValueError(
            f"La carpeta de trabajo '{destino.name}' ya existe y no está vacía: usa otro exp-id "
            "o bórrala a mano (no se borra nada automáticamente)"
        )

    copias: list[tuple[str, str]] = []
    if smoke_imagenes is not None:
        cocos["train"] = subconjunto_smoke(cocos["train"], smoke_imagenes, semilla)
    if repeticiones_svodk > 1:
        cocos["train"], copias = _repetir(cocos["train"], prefijo_svodk, repeticiones_svodk)

    recuentos = {}
    for carpeta, coco in cocos.items():
        origen, salida = dataset / carpeta, destino / carpeta
        salida.mkdir(parents=True, exist_ok=True)
        propias = {n for n, _ in copias} if carpeta == "train" else set()
        for imagen in coco["images"]:
            nombre = imagen["file_name"]
            if nombre not in propias:  # las repeticiones se copian aparte, desde su original
                shutil.copy2(origen / nombre, salida / nombre)
        if carpeta == "train":
            for nuevo, original in copias:
                shutil.copy2(origen / original, salida / nuevo)
            (salida / NOMBRE_JSON).write_text(
                json.dumps(coco, ensure_ascii=False), encoding="utf-8"
            )
        else:
            shutil.copy2(origen / NOMBRE_JSON, salida / NOMBRE_JSON)  # idéntico al original
        recuentos[carpeta] = _contar(coco, prefijo_svodk)
    return recuentos
