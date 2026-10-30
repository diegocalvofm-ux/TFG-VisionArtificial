"""Importación de datasets públicos (COCO de Roboflow) a metadata.csv y al dataset de la etapa 1.

Qué hace, en lenguaje llano: lee los ZIP de ``data/raw`` sin tocarlos, traduce las clases de
cada ZIP a un destino (``lechuga``, ``kern``, ``tipburn`` o ``ignorar``), reparte las imágenes
en train / valid / test, escribe una fila por imagen en ``metadata.csv`` y exporta a
``data/processed/etapa1_v1`` las imágenes con una sola clase, ``lechuga``. Las columnas y los
valores válidos de metadata.csv salen siempre de ``src/data/schema.py``.

IMPORTANTE sobre ``lechuga_id``: en datos públicos NO es una lechuga física. Es un *grupo de
partición*: el conjunto de imágenes que no deben separarse entre train, valid y test (por
ejemplo, fotogramas casi idénticos). Se saca del nombre de archivo con ``grupo_regex``; si la
fuente no tiene regex, la columna va vacía y cada imagen cuenta como su propio grupo (con el
aviso que emite ``group_split``: fotos de la misma planta podrían caer en particiones distintas).

Reparto temporal (``particion_temporal`` en el YAML de una fuente): cuando una fuente es UN SOLO
VÍDEO, los fotogramas contiguos son casi iguales y repartirlos al azar metería en valid y test
fotos casi idénticas a las de train. Por eso se ordenan por el tiempo que lleva su nombre, los
primeros van a train, los siguientes a valid y (si se piden tres fracciones) los últimos a test, y
se descartan los fotogramas a menos de ``margen_s`` segundos de cada corte para que haya un hueco
entre particiones.
"""

import itertools
import json
import math
import random
import re
import shutil
import warnings
import zipfile
from collections import Counter
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.config import load_config
from src.data.coco import (
    CARPETAS,
    area_poligonos,
    bbox_desde_poligonos,
    clases_con_anotaciones,
    escribir_coco,
    leer_anotaciones_zip,
    nombres_de_categorias,
    poligono_valido,
)
from src.data.schema import GROUPS, METADATA_COLUMNS, NO_VISIBLE, validate_metadata
from src.data.split import _repartir, _validar_fracciones, group_split
from src.evaluation.aggregate import aggregate
from src.pipeline.imagenes import guardar_imagen, leer_imagen

# Destinos posibles de cada clase del ZIP. Solo "lechuga" entra en el dataset de la etapa 1.
DESTINOS = ("lechuga", "kern", "tipburn", "ignorar")
PAPELES = ("train_only", "split")

# group_split llama "val" a la partición de validación; Roboflow y el dataset exportado "valid".
PARTICION_SALIDA = {"train": "train", "val": "valid", "test": "test"}
PARTICIONES_EXPORTADAS = tuple(PARTICION_SALIDA.values())

# Motivo con el que el informe anota los fotogramas descartados por estar cerca de un corte.
MOTIVO_MARGEN = "margen temporal"

# Categoría única del dataset de la etapa 1 (el id 1 deja el 0 libre para el fondo).
CATEGORIA_LECHUGA = {"id": 1, "name": "lechuga"}

_NOMBRE_FUENTE = re.compile(r"[A-Za-z0-9_-]+")
# Lo único que puede haber en una carpeta de salida para que se pueda borrar y regenerar.
_CONTENIDO_PROPIO = frozenset((*PARTICIONES_EXPORTADAS, "informe.json"))


@dataclass(frozen=True)
class Temporal:
    """Reparto temporal de una fuente que es un solo vídeo (bloque ``particion_temporal``)."""

    patron: re.Pattern  # 4 grupos: horas, minutos, segundos y centésimas del nombre
    # (train, valid) con un solo corte, o (train, valid, test) con dos; en orden temporal.
    fracciones: tuple[float, ...]
    margen_s: float = 5  # se descartan los fotogramas a menos de esto de cada corte


@dataclass(frozen=True)
class Fuente:
    """Configuración de una fuente pública (un ``configs/datasets/<fuente>.yaml``)."""

    nombre: str
    licencia: str
    zip: Path
    clases: dict[str, str]
    papel: str
    fracciones: tuple[float, float, float] | None = None  # (train, valid, test)
    grupo_regex: re.Pattern | None = None
    excluir: tuple[str, ...] = ()
    particion_temporal: Temporal | None = None


@dataclass(frozen=True)
class Etapa:
    """Configuración del dataset de la etapa 1 (``configs/datasets/etapa1_v1.yaml``)."""

    fuentes: tuple[str, ...]
    semilla: int
    salida: Path
    carpeta_fuentes: Path  # donde están los YAML de cada fuente


@dataclass
class Imagen:
    """Una imagen de un ZIP con lo necesario para metadata.csv y para la exportación."""

    image_id: str
    fuente: str
    licencia: str
    carpeta: str  # carpeta original del ZIP (train/valid/test); solo informativa
    archivo_original: str  # ruta dentro del ZIP, p. ej. "train/foto.jpg"
    nombre: str  # nombre del archivo, sin carpeta
    ancho: int
    alto: int
    clases_presentes: tuple[str, ...]  # clases originales con anotaciones, ordenadas
    lechuga_id: str  # grupo de partición ("" si la fuente no tiene grupo_regex)
    excluida: bool = False
    anotaciones: list[list[list[float]]] = field(default_factory=list)  # polígonos de cada lechuga
    descartadas: int = 0  # anotaciones de lechuga descartadas por polígono degenerado
    # Solo con reparto temporal: segundos del fotograma, su partición y si se descartó por margen.
    tiempo_s: float | None = None
    particion_temporal: str | None = None
    descartada_margen: bool = False

    @property
    def motivo(self) -> str | None:
        """Por qué no se exporta (``None`` si se exporta)."""
        if self.excluida:
            return "excluir_imagenes"
        if self.descartada_margen:
            return MOTIVO_MARGEN
        if not self.anotaciones:
            return "sin_lechuga"
        return None


# --- Configuración ----------------------------------------------------------------------


def cargar_fuente(ruta: str | Path, raiz: str | Path) -> Fuente:
    """Lee y valida el YAML de una fuente; la ruta del ZIP es relativa a ``raiz``.

    En lenguaje llano: comprueba que el YAML tiene todo lo necesario y que los valores son
    válidos (destinos, papel, fracciones, regex) y falla con un mensaje que nombra la fuente y
    el problema si algo no cuadra.
    """
    ruta = Path(ruta)
    config = load_config(ruta)
    nombre = config.get("fuente")
    if not isinstance(nombre, str) or not _NOMBRE_FUENTE.fullmatch(nombre):
        raise ValueError(f"{ruta.name}: 'fuente' debe ser un nombre sin espacios (letras, - y _)")
    if nombre != ruta.stem:
        raise ValueError(
            f"{ruta.name}: 'fuente' ({nombre}) debe coincidir con el nombre del archivo"
        )

    def falta(clave):
        return ValueError(f"Fuente '{nombre}': falta '{clave}' en {ruta.name}")

    for clave in ("licencia", "zip", "clases", "papel"):
        if config.get(clave) in (None, "", {}):
            raise falta(clave)
    clases = config["clases"]
    if not isinstance(clases, dict):
        raise ValueError(f"Fuente '{nombre}': 'clases' debe ser clase_original: destino")
    for clase, destino in clases.items():
        if destino not in DESTINOS:
            raise ValueError(
                f"Fuente '{nombre}': destino '{destino}' no válido para la clase '{clase}' "
                f"(válidos: {', '.join(DESTINOS)})"
            )
    papel = config["papel"]
    if papel not in PAPELES:
        raise ValueError(
            f"Fuente '{nombre}': papel '{papel}' no válido (válidos: {', '.join(PAPELES)})"
        )

    temporal = None
    if config.get("particion_temporal") is not None:
        temporal = _cargar_temporal(
            nombre, papel, config["particion_temporal"], bool(config.get("grupo_regex"))
        )

    fracciones = None
    if papel == "split" and temporal is None:  # con reparto temporal, las fracciones van dentro
        crudas = config.get("fracciones")
        if not isinstance(crudas, dict) or set(crudas) != {"train", "valid", "test"}:
            raise ValueError(
                f"Fuente '{nombre}': con papel 'split' hacen falta 'fracciones' con las claves "
                "train, valid y test"
            )
        fracciones = (crudas["train"], crudas["valid"], crudas["test"])
        try:
            _validar_fracciones(fracciones)
        except ValueError as error:
            raise ValueError(f"Fuente '{nombre}': {error}") from error

    regex = None
    if config.get("grupo_regex"):
        try:
            regex = re.compile(config["grupo_regex"])
        except re.error as error:
            raise ValueError(f"Fuente '{nombre}': grupo_regex no válida: {error}") from error

    ruta_zip = Path(raiz) / config["zip"]
    if not ruta_zip.is_file():
        raise FileNotFoundError(f"Fuente '{nombre}': no se encuentra el ZIP {config['zip']}")

    return Fuente(
        nombre=nombre,
        licencia=str(config["licencia"]),
        zip=ruta_zip,
        clases=dict(clases),
        papel=papel,
        fracciones=fracciones,
        grupo_regex=regex,
        excluir=tuple(config.get("excluir_imagenes") or ()),
        particion_temporal=temporal,
    )


def _cargar_temporal(nombre: str, papel: str, cruda, hay_grupo_regex: bool) -> Temporal:
    """Valida el bloque ``particion_temporal`` de una fuente y lo convierte en ``Temporal``.

    En lenguaje llano: comprueba que el reparto temporal tiene sentido (solo con papel
    ``split``, sin ``grupo_regex`` a la vez, patrón con 4 grupos, fracciones que suman 1 y un
    margen no negativo) y falla con un mensaje que nombra la fuente y el bloque. Las fracciones
    son ``train`` y ``valid`` (un corte; ``test`` queda sin imágenes) o ``train``, ``valid`` y
    ``test`` (dos cortes).
    """

    def error(texto: str) -> ValueError:
        return ValueError(f"Fuente '{nombre}': particion_temporal: {texto}")

    if papel != "split":
        raise error(f"solo se puede usar con papel 'split' (la fuente tiene '{papel}')")
    if hay_grupo_regex:
        raise error("no puede combinarse con grupo_regex (son dos criterios de reparto distintos)")
    if not isinstance(cruda, dict):
        raise error("debe ser un bloque con patron, fracciones y margen_s")
    desconocidas = sorted(set(cruda) - {"patron", "fracciones", "margen_s"})
    if desconocidas:
        raise error(f"claves desconocidas: {', '.join(map(str, desconocidas))}")

    try:
        patron = re.compile(str(cruda.get("patron") or ""))
    except re.error as fallo:
        raise error(f"patron no válido: {fallo}") from fallo
    if not cruda.get("patron") or patron.groups != 4:
        raise error(
            "'patron' debe tener 4 grupos (horas, minutos, segundos y centésimas), p. ej. "
            r"frame(\d+)-(\d+)-(\d+)-(\d+)_"
        )

    crudas = cruda.get("fracciones")
    if not isinstance(crudas, dict) or set(crudas) not in (
        {"train", "valid"},
        {"train", "valid", "test"},
    ):
        raise error("'fracciones' necesita las claves train y valid (y, opcional, test)")
    fracciones = tuple(crudas[p] for p in PARTICIONES_EXPORTADAS[: len(crudas)])
    if any(isinstance(f, bool) or not isinstance(f, int | float) or f <= 0 for f in fracciones):
        raise error(f"las fracciones deben ser números positivos: {fracciones}")
    if not math.isclose(sum(fracciones), 1.0, abs_tol=1e-9):
        raise error(f"las fracciones deben sumar 1 y suman {sum(fracciones)}: {fracciones}")

    margen = cruda.get("margen_s", 5)
    if isinstance(margen, bool) or not isinstance(margen, int | float) or margen < 0:
        raise error(f"'margen_s' debe ser un número mayor o igual que 0 (llegó {margen!r})")
    return Temporal(patron=patron, fracciones=fracciones, margen_s=margen)


def cargar_etapa(ruta: str | Path, raiz: str | Path) -> Etapa:
    """Lee y valida el YAML del dataset (fuentes, semilla y carpeta de salida)."""
    ruta = Path(ruta)
    config = load_config(ruta)
    fuentes = config.get("fuentes")
    if not isinstance(fuentes, list) or not fuentes:
        raise ValueError(f"{ruta.name}: 'fuentes' debe ser una lista con al menos una fuente")
    if len(set(fuentes)) != len(fuentes):
        raise ValueError(f"{ruta.name}: hay fuentes repetidas en 'fuentes'")
    semilla = config.get("semilla")
    if isinstance(semilla, bool) or not isinstance(semilla, int):
        raise ValueError(f"{ruta.name}: 'semilla' debe ser un número entero")
    if not config.get("salida"):
        raise ValueError(f"{ruta.name}: falta 'salida'")
    return Etapa(
        fuentes=tuple(fuentes),
        semilla=semilla,
        salida=Path(raiz) / config["salida"],
        carpeta_fuentes=ruta.parent,
    )


def cargar_fuentes(etapa: Etapa, raiz: str | Path) -> list[Fuente]:
    """Carga el YAML de cada fuente de la etapa (``<carpeta_fuentes>/<fuente>.yaml``)."""
    return [cargar_fuente(etapa.carpeta_fuentes / f"{n}.yaml", raiz) for n in etapa.fuentes]


# --- Lectura de un ZIP ------------------------------------------------------------------


def grupo_de(nombre: str, regex: re.Pattern | None, fuente: str) -> str:
    """Devuelve el ``lechuga_id`` (grupo de partición) de una imagen a partir de su nombre.

    En lenguaje llano: con ``grupo_regex`` se busca en el nombre del archivo; el grupo 1 de la
    regex (o toda la coincidencia si no tiene grupos) con el prefijo ``<fuente>_`` es el id.
    Sin regex devuelve ``""``: la imagen será su propio grupo. Si el nombre no encaja, error.
    """
    if regex is None:
        return ""
    coincidencia = regex.search(nombre)
    if coincidencia is None:
        raise ValueError(
            f"Fuente '{fuente}': el nombre '{nombre}' no encaja con grupo_regex '{regex.pattern}'"
        )
    grupo = coincidencia.group(1) if regex.groups else coincidencia.group(0)
    return f"{fuente}_{grupo}"


def tiempo_de(nombre: str, patron: re.Pattern, fuente: str) -> float:
    """Segundos que marca el nombre de un fotograma, leídos con ``patron`` (4 grupos).

    En lenguaje llano: de ``frame1-02-03-45_...`` saca horas=1, minutos=2, segundos=3 y
    centésimas=45, y los junta: ``1*3600 + 2*60 + 3 + 45/100 = 3723.45``. Si el nombre no
    encaja con el patrón lanza ``ValueError`` citando el nombre.
    """
    coincidencia = patron.search(nombre)
    if coincidencia is None:
        raise ValueError(
            f"Fuente '{fuente}': el nombre '{nombre}' no encaja con el patron de "
            f"particion_temporal '{patron.pattern}'"
        )
    horas, minutos, segundos, centesimas = (int(g) for g in coincidencia.groups())
    return horas * 3600 + minutos * 60 + segundos + centesimas / 100


def asignar_particion_temporal(imagenes: list[Imagen], temporal: Temporal, fuente: str) -> None:
    """Reparte los fotogramas de un vídeo por orden de tiempo y marca los de margen.

    En lenguaje llano: se ordenan las imágenes (no excluidas) por su tiempo y se parten en
    trozos consecutivos con las fracciones dadas (por número de imágenes, con el mismo redondeo
    que ``group_split``): las primeras son train, las siguientes valid y, si hay tres fracciones,
    las últimas test (con dos, test queda sin imágenes). Cada corte cae a mitad de camino entre
    la última imagen de un trozo y la primera del siguiente, y se marcan como
    ``descartada_margen`` los fotogramas a menos de ``margen_s`` segundos de un corte (a los dos
    lados). Así entre dos particiones queda un hueco de al menos 2·margen_s. No usa semilla: el
    resultado solo depende de los nombres, no del orden dentro del ZIP.
    Modifica las imágenes en el sitio. Lanza ``ValueError`` si algún nombre no encaja con el
    patrón, si hay menos imágenes que trozos o si una partición se queda vacía tras el margen.
    """
    secuencia = [imagen for imagen in imagenes if not imagen.excluida]
    for imagen in secuencia:
        imagen.tiempo_s = tiempo_de(imagen.nombre, temporal.patron, fuente)
    secuencia.sort(key=lambda imagen: (imagen.tiempo_s, imagen.nombre))
    usadas = PARTICIONES_EXPORTADAS[: len(temporal.fracciones)]
    if len(secuencia) < len(usadas):
        raise ValueError(
            f"Fuente '{fuente}': el reparto temporal necesita al menos {len(usadas)} imágenes "
            f"y hay {len(secuencia)}"
        )

    # Posición donde acaba cada trozo salvo el último (un solo corte con dos fracciones).
    finales = list(itertools.accumulate(_repartir(len(secuencia), temporal.fracciones)))[:-1]
    cortes = [(secuencia[i - 1].tiempo_s + secuencia[i].tiempo_s) / 2 for i in finales]
    for posicion, imagen in enumerate(secuencia):
        trozo = sum(posicion >= fin for fin in finales)
        imagen.particion_temporal = usadas[trozo]
        imagen.descartada_margen = any(
            abs(imagen.tiempo_s - corte) < temporal.margen_s for corte in cortes
        )

    for particion in usadas:
        if not any(
            i.particion_temporal == particion and not i.descartada_margen for i in secuencia
        ):
            raise ValueError(
                f"Fuente '{fuente}': la partición '{particion}' queda vacía al descartar los "
                f"fotogramas a menos de {temporal.margen_s} s de cada corte (margen_s demasiado "
                "grande o pocas imágenes)"
            )


def leer_fuente(fuente: Fuente) -> list[Imagen]:
    """Lee un ZIP y devuelve sus imágenes (también las excluidas, marcadas) en orden estable.

    En lenguaje llano: recorre las carpetas train, valid y test, ordena las imágenes por
    carpeta y nombre y les da el id ``<fuente>_<nnnn>`` contando desde 1. Las imágenes excluidas
    también gastan número, así excluir una no cambia el id de las demás. De cada anotación se
    mira su clase: la del YAML decide si es lechuga (se guarda su polígono) o no. Una clase con
    anotaciones y sin mapeo en el YAML lanza ``ValueError``. El ZIP nunca se modifica.
    """
    with zipfile.ZipFile(fuente.zip) as zf:
        por_carpeta = leer_anotaciones_zip(zf)
        contenido = set(zf.namelist())

    totales: Counter = Counter()
    existentes: set[str] = set()
    for coco in por_carpeta.values():
        totales.update(clases_con_anotaciones(coco))
        existentes.update(nombres_de_categorias(coco).values())
    sin_mapeo = sorted(c for c in totales if c not in fuente.clases)
    if sin_mapeo:
        detalle = ", ".join(f"'{c}' ({totales[c]} anotaciones)" for c in sin_mapeo)
        raise ValueError(
            f"Fuente '{fuente.nombre}': clase(s) del ZIP sin mapeo en el YAML: {detalle}. "
            f"Añádelas a 'clases' con un destino de: {', '.join(DESTINOS)}"
        )
    for clase in fuente.clases:
        if clase not in existentes:
            warnings.warn(
                f"Fuente '{fuente.nombre}': la clase '{clase}' del YAML no está en el ZIP",
                UserWarning,
                stacklevel=2,
            )

    imagenes: list[Imagen] = []
    usadas: set[str] = set()
    for carpeta in CARPETAS:
        coco = por_carpeta.get(carpeta)
        if coco is None:
            continue
        categorias = nombres_de_categorias(coco)
        anotaciones_de: dict[int, list[dict]] = {}
        for anotacion in coco["annotations"]:
            anotaciones_de.setdefault(anotacion["image_id"], []).append(anotacion)

        for info in sorted(coco["images"], key=lambda i: i["file_name"]):
            nombre = info["file_name"]
            ruta_zip = f"{carpeta}/{nombre}"
            if ruta_zip not in contenido:
                raise ValueError(
                    f"Fuente '{fuente.nombre}': el JSON cita '{ruta_zip}' pero no está en el ZIP"
                )
            excluida = False
            for entrada in fuente.excluir:
                if entrada in (nombre, ruta_zip):
                    excluida = True
                    usadas.add(entrada)

            presentes: set[str] = set()
            poligonos_lechuga: list[list[list[float]]] = []
            descartadas = 0
            for anotacion in anotaciones_de.get(info["id"], []):
                clase = categorias[anotacion["category_id"]]
                presentes.add(clase)
                if fuente.clases[clase] != "lechuga":
                    continue
                segmentacion = anotacion["segmentation"]
                if not isinstance(segmentacion, list):
                    raise ValueError(
                        f"Fuente '{fuente.nombre}': '{ruta_zip}' tiene una segmentación que no "
                        "es una lista de polígonos (¿RLE?); no está soportada"
                    )
                validos = [p for p in segmentacion if poligono_valido(p)]
                if validos:
                    poligonos_lechuga.append(validos)
                else:
                    descartadas += 1

            n = len(imagenes) + 1
            imagenes.append(
                Imagen(
                    image_id=f"{fuente.nombre}_{n:04d}",
                    fuente=fuente.nombre,
                    licencia=fuente.licencia,
                    carpeta=carpeta,
                    archivo_original=ruta_zip,
                    nombre=nombre,
                    ancho=info["width"],
                    alto=info["height"],
                    clases_presentes=tuple(sorted(presentes)),
                    lechuga_id=""
                    if excluida
                    else grupo_de(nombre, fuente.grupo_regex, fuente.nombre),
                    excluida=excluida,
                    anotaciones=poligonos_lechuga,
                    descartadas=descartadas,
                )
            )

    sin_coincidencia = [e for e in fuente.excluir if e not in usadas]
    if sin_coincidencia:
        raise ValueError(
            f"Fuente '{fuente.nombre}': excluir_imagenes no coincide con ninguna imagen del ZIP: "
            f"{', '.join(sin_coincidencia)}"
        )
    if fuente.particion_temporal is not None:
        asignar_particion_temporal(imagenes, fuente.particion_temporal, fuente.nombre)
    return imagenes


# --- metadata.csv -----------------------------------------------------------------------


def filas_metadata(imagenes: list[Imagen], tabla: dict) -> pd.DataFrame:
    """Una fila de ``metadata.csv`` por imagen que forma parte del dataset (con todas las columnas).

    No hay fila para las imágenes excluidas por el YAML ni para las descartadas por margen
    temporal: no forman parte del dataset.

    En lenguaje llano: los seis grupos de calidad van en ``NO_VISIBLE`` (las fotos públicas no
    están etiquetadas con nuestros grupos), ``salida_global`` sale de la regla de agregación
    existente con esos seis valores y las columnas de captura quedan vacías.
    ``etiqueta_original`` junta, separadas por ``;``, las clases que traía el ZIP en esa imagen.
    """
    nivel = aggregate(dict.fromkeys(GROUPS, NO_VISIBLE), tabla).level
    filas = []
    for imagen in imagenes:
        if imagen.excluida or imagen.descartada_margen:
            continue
        fila = dict.fromkeys(METADATA_COLUMNS, "")
        fila.update(
            image_id=imagen.image_id,
            fuente=imagen.fuente,
            licencia=imagen.licencia,
            etiqueta_original=";".join(imagen.clases_presentes),
            lechuga_id=imagen.lechuga_id,
            salida_global=nivel,
            archivo_original=imagen.archivo_original,
        )
        fila.update(dict.fromkeys(GROUPS, NO_VISIBLE))
        filas.append(fila)
    return pd.DataFrame(filas, columns=list(METADATA_COLUMNS))


def leer_metadata(ruta: str | Path) -> pd.DataFrame | None:
    """Lee metadata.csv como texto (sin convertir vacíos a NaN); ``None`` si no existe."""
    ruta = Path(ruta)
    if not ruta.is_file():
        return None
    return pd.read_csv(ruta, dtype=str, keep_default_na=False, encoding="utf-8")


def fusionar_metadata(
    existente: pd.DataFrame | None, nuevas: pd.DataFrame, fuentes: list[str]
) -> pd.DataFrame:
    """Quita de la tabla existente las filas de ``fuentes`` y añade las nuevas.

    En lenguaje llano: volver a importar una fuente sustituye sus filas (no las duplica) y deja
    intactas las de las demás fuentes. No modifica ninguna de las dos tablas de entrada.
    """
    if existente is None or existente.empty:
        return nuevas.copy()
    if "fuente" not in existente.columns:
        raise ValueError("El metadata.csv existente no tiene la columna 'fuente'")
    conservadas = existente[~existente["fuente"].isin(fuentes)]
    return pd.concat([conservadas, nuevas], ignore_index=True)


def comprobar_metadata(df: pd.DataFrame) -> None:
    """Lanza ``ValueError`` con los primeros problemas si la tabla no cumple el esquema."""
    errores = validate_metadata(df)
    if errores:
        resumen = "\n  ".join(errores[:10])
        extra = f"\n  ... y {len(errores) - 10} más" if len(errores) > 10 else ""
        raise ValueError(
            f"metadata.csv no cumple el esquema ({len(errores)} problemas):\n  {resumen}{extra}"
        )


def escribir_metadata(df: pd.DataFrame, ruta: str | Path) -> None:
    """Valida la tabla y la escribe en CSV (utf-8); si no es válida no escribe nada."""
    comprobar_metadata(df)
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ruta, index=False, encoding="utf-8", lineterminator="\n")


# --- Partición --------------------------------------------------------------------------


def repartir(
    imagenes: list[Imagen], fuentes: dict[str, Fuente], semilla: int
) -> tuple[dict[str, str], list[str]]:
    """Asigna cada imagen no excluida a train, valid o test. Devuelve (particiones, avisos).

    En lenguaje llano: las fuentes ``train_only`` van enteras a train. Las ``split`` se reparten
    con ``group_split`` (todas las imágenes de un mismo grupo ``lechuga_id`` en la misma
    partición), cada fuente por separado y con la misma semilla, así que el resultado es siempre
    el mismo. Se reparten también las imágenes sin lechuga, para que su presencia no cambie el
    reparto del resto. Los avisos de ``group_split`` se devuelven para ponerlos en el informe.

    Las fuentes con ``particion_temporal`` ya vienen repartidas por tiempo desde ``leer_fuente``
    (no usan la semilla); aquí solo se recogen, sin los fotogramas descartados por margen.
    """
    particiones: dict[str, str] = {}
    avisos: list[str] = []
    for nombre, fuente in fuentes.items():
        propias = [i for i in imagenes if i.fuente == nombre and not i.excluida]
        if not propias:
            continue
        if fuente.papel == "train_only":
            particiones.update({i.image_id: "train" for i in propias})
            continue
        if fuente.particion_temporal is not None:
            # Ya repartidas por tiempo en leer_fuente (sin semilla); las del margen no entran.
            particiones.update(
                {i.image_id: i.particion_temporal for i in propias if not i.descartada_margen}
            )
            continue
        df = pd.DataFrame(
            {
                "image_id": [i.image_id for i in propias],
                "lechuga_id": [i.lechuga_id for i in propias],
            }
        )
        with warnings.catch_warnings(record=True) as capturados:
            warnings.simplefilter("always")
            try:
                reparto = group_split(df, "lechuga_id", fuente.fracciones, semilla)
            except ValueError as error:
                raise ValueError(f"Fuente '{nombre}': {error}") from error
        avisos += [
            f"{nombre}: {c.message}" for c in capturados if issubclass(c.category, UserWarning)
        ]
        for image_id, particion in zip(reparto["image_id"], reparto["split"], strict=True):
            particiones[image_id] = PARTICION_SALIDA[particion]
    return particiones, avisos


# --- Exportación ------------------------------------------------------------------------


def preparar_salida(salida: Path) -> None:
    """Borra la salida anterior y crea las carpetas vacías train, valid y test.

    En lenguaje llano: como esto BORRA una carpeta, solo se atreve si no existe o si contiene
    únicamente lo que genera este importador (train, valid, test e informe.json). Si hay
    cualquier otra cosa, lanza ``ValueError`` y no toca nada.
    """
    if salida.exists():
        if not salida.is_dir():
            raise ValueError(f"La salida {salida.name} existe y no es una carpeta")
        ajenos = sorted(p.name for p in salida.iterdir() if p.name not in _CONTENIDO_PROPIO)
        if ajenos:
            raise ValueError(
                f"No se borra la salida {salida.name}: contiene archivos que no son del "
                f"importador ({', '.join(ajenos)}). Muévelos o elige otra carpeta de salida."
            )
        shutil.rmtree(salida)
    for particion in PARTICIONES_EXPORTADAS:
        (salida / particion).mkdir(parents=True)


def exportar(
    imagenes: list[Imagen], particiones: dict[str, str], fuentes: dict[str, Fuente], salida: Path
) -> None:
    """Escribe el dataset de la etapa 1: imágenes con su id corto y un COCO por partición.

    En lenguaje llano: por cada imagen con alguna lechuga se copia el archivo del ZIP a
    ``<salida>/<partición>/<image_id>.<ext>`` y se anota cada lechuga con su polígono, su caja
    (la que envuelve al polígono) y su área (calculada con el polígono). Todas las lechugas son
    la categoría ``lechuga`` (id 1). Las imágenes excluidas o sin lechuga no se exportan.
    """
    salida = Path(salida)
    preparar_salida(salida)
    datos = {p: ([], []) for p in PARTICIONES_EXPORTADAS}  # partición -> (imágenes, anotaciones)
    with ExitStack() as pila:
        zips: dict[str, zipfile.ZipFile] = {}
        for imagen in imagenes:
            if imagen.motivo is not None:
                continue
            if imagen.fuente not in zips:
                zips[imagen.fuente] = pila.enter_context(
                    zipfile.ZipFile(fuentes[imagen.fuente].zip)
                )
            particion = particiones[imagen.image_id]
            nombre = f"{imagen.image_id}{Path(imagen.nombre).suffix.lower() or '.jpg'}"
            (salida / particion / nombre).write_bytes(
                zips[imagen.fuente].read(imagen.archivo_original)
            )

            imgs, anns = datos[particion]
            id_imagen = len(imgs) + 1
            imgs.append(
                {"id": id_imagen, "file_name": nombre, "width": imagen.ancho, "height": imagen.alto}
            )
            for poligonos in imagen.anotaciones:
                anns.append(
                    {
                        "id": len(anns) + 1,
                        "image_id": id_imagen,
                        "category_id": CATEGORIA_LECHUGA["id"],
                        "bbox": list(bbox_desde_poligonos(poligonos)),
                        "area": area_poligonos(poligonos),
                        "iscrowd": 0,
                        "segmentation": poligonos,
                    }
                )
    for particion, (imgs, anns) in datos.items():
        escribir_coco(
            salida / particion / "_annotations.coco.json", imgs, anns, [dict(CATEGORIA_LECHUGA)]
        )


# --- Informe ----------------------------------------------------------------------------


def construir_informe(
    imagenes: list[Imagen],
    particiones: dict[str, str],
    semilla: int,
    avisos: list[str],
    fuentes_config: dict[str, Fuente] | None = None,
) -> dict:
    """Resume qué se ha exportado y qué no, y por qué (listo para guardar como JSON).

    Incluye imágenes y anotaciones exportadas por fuente y partición, las imágenes sin ninguna
    lechuga, las excluidas con su motivo (``excluir_imagenes`` o ``margen temporal``), las
    anotaciones descartadas por polígono degenerado y los avisos de la partición. Si se pasa la
    configuración de las fuentes, las que tienen reparto temporal añaden su bloque
    ``particion_temporal``: margen, fotogramas descartados y, por partición, nº de imágenes y
    rango de tiempo (``desde_s`` y ``hasta_s``) de las que se quedan.
    """
    vacio = {"imagenes": 0, "anotaciones": 0}
    fuentes: dict[str, dict] = {}
    totales = {p: dict(vacio) for p in PARTICIONES_EXPORTADAS}
    sin_lechuga, excluidas, descartadas = [], [], 0
    for imagen in imagenes:
        por_particion = fuentes.setdefault(
            imagen.fuente, {"particiones": {p: dict(vacio) for p in PARTICIONES_EXPORTADAS}}
        )["particiones"]
        descartadas += imagen.descartadas
        if imagen.motivo in ("excluir_imagenes", MOTIVO_MARGEN):
            excluidas.append(
                {"fuente": imagen.fuente, "archivo_original": imagen.archivo_original,
                 "motivo": imagen.motivo}
            )  # fmt: skip
        elif imagen.motivo == "sin_lechuga":
            sin_lechuga.append(
                {"fuente": imagen.fuente, "image_id": imagen.image_id,
                 "archivo_original": imagen.archivo_original, "motivo": imagen.motivo}
            )  # fmt: skip
        else:
            for destino in (
                por_particion[particiones[imagen.image_id]],
                totales[particiones[imagen.image_id]],
            ):
                destino["imagenes"] += 1
                destino["anotaciones"] += len(imagen.anotaciones)

    for nombre, config in (fuentes_config or {}).items():
        if config.particion_temporal is None or nombre not in fuentes:
            continue
        propias = [i for i in imagenes if i.fuente == nombre and i.image_id in particiones]
        por_particion = {}
        for particion in PARTICIONES_EXPORTADAS[: len(config.particion_temporal.fracciones)]:
            tiempos = [i.tiempo_s for i in propias if particiones[i.image_id] == particion]
            por_particion[particion] = {
                "imagenes": len(tiempos),
                "desde_s": min(tiempos) if tiempos else None,
                "hasta_s": max(tiempos) if tiempos else None,
            }
        fuentes[nombre]["particion_temporal"] = {
            "margen_s": config.particion_temporal.margen_s,
            "descartadas_margen": sum(
                1 for i in imagenes if i.fuente == nombre and i.descartada_margen
            ),
            "particiones": por_particion,
        }
    return {
        "semilla": semilla,
        "fuentes": fuentes,
        "totales": totales,
        "sin_lechuga": sin_lechuga,
        "excluidas": excluidas,
        "anotaciones_descartadas": descartadas,
        "avisos": list(avisos),
    }


def formatear_informe(informe: dict) -> str:
    """Texto legible del informe para la consola."""
    lineas = [f"Informe de la importación (semilla {informe['semilla']})"]
    for nombre, datos in informe["fuentes"].items():
        lineas.append(f"  {nombre}:")
        for particion, c in datos["particiones"].items():
            lineas.append(
                f"    {particion}: {c['imagenes']} imágenes, {c['anotaciones']} anotaciones"
            )
        temporal = datos.get("particion_temporal")
        if temporal:
            lineas.append(
                f"    reparto temporal (margen {temporal['margen_s']} s, descartadas por "
                f"{MOTIVO_MARGEN}: {temporal['descartadas_margen']}):"
            )
            for particion, c in temporal["particiones"].items():
                rango = (
                    f"de {c['desde_s']:.2f} s a {c['hasta_s']:.2f} s" if c["imagenes"] else "vacía"
                )
                lineas.append(f"      {particion}: {c['imagenes']} imágenes, {rango}")
    lineas.append("  total:")
    for particion, c in informe["totales"].items():
        lineas.append(f"    {particion}: {c['imagenes']} imágenes, {c['anotaciones']} anotaciones")
    lineas.append(f"  imágenes sin ninguna lechuga (no exportadas): {len(informe['sin_lechuga'])}")
    por_margen = sum(1 for e in informe["excluidas"] if e["motivo"] == MOTIVO_MARGEN)
    lineas.append(f"  imágenes excluidas por el YAML: {len(informe['excluidas']) - por_margen}")
    lineas.append(f"  imágenes descartadas por {MOTIVO_MARGEN}: {por_margen}")
    lineas.append(
        f"  anotaciones descartadas por polígono degenerado: {informe['anotaciones_descartadas']}"
    )
    lineas += [f"  AVISO: {aviso}" for aviso in informe["avisos"]]
    return "\n".join(lineas)


# --- Orquestación -----------------------------------------------------------------------


def importar(
    ruta_etapa: str | Path,
    *,
    raiz: str | Path,
    metadata: str | Path,
    tabla: dict,
    semilla: int | None = None,
) -> dict:
    """Importa todas las fuentes de la etapa: metadata.csv, dataset exportado e informe.

    En lenguaje llano: lee los ZIP, reparte, construye la tabla de metadatos fusionada con la que
    ya hubiera y la valida. Solo si todo eso sale bien exporta (borrando la salida anterior) y, por
    último, escribe ``informe.json`` y ``metadata.csv``. Así un error no deja el CSV a medias.
    Devuelve el informe. ``semilla`` sustituye a la del YAML si se da.
    """
    etapa = cargar_etapa(ruta_etapa, raiz)
    fuentes = cargar_fuentes(etapa, raiz)
    semilla = etapa.semilla if semilla is None else semilla

    imagenes: list[Imagen] = []
    for fuente in fuentes:
        imagenes += leer_fuente(fuente)
    por_nombre = {f.nombre: f for f in fuentes}
    particiones, avisos = repartir(imagenes, por_nombre, semilla)

    total = fusionar_metadata(
        leer_metadata(metadata), filas_metadata(imagenes, tabla), [f.nombre for f in fuentes]
    )
    comprobar_metadata(total)

    exportar(imagenes, particiones, por_nombre, etapa.salida)
    informe = construir_informe(imagenes, particiones, semilla, avisos, por_nombre)
    (etapa.salida / "informe.json").write_text(
        json.dumps(informe, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    escribir_metadata(total, metadata)
    return informe


# --- Listado de un ZIP (no escribe nada) ------------------------------------------------


def describir_zip(ruta: str | Path, ejemplos: int = 20) -> str:
    """Texto con las clases del ZIP y su nº de anotaciones, las imágenes por carpeta y ejemplos.

    En lenguaje llano: sirve para decidir el mapeo de clases y el ``grupo_regex`` antes de
    escribir el YAML. Solo lee el ZIP. Los ejemplos de nombres se reparten a lo largo de la
    lista ordenada de imágenes (no son solo los primeros).
    """
    with zipfile.ZipFile(ruta) as zf:
        por_carpeta = leer_anotaciones_zip(zf)
    clases: Counter = Counter()
    todas = set()
    nombres: list[str] = []
    lineas = []
    for coco in por_carpeta.values():
        clases.update(clases_con_anotaciones(coco))
        todas.update(nombres_de_categorias(coco).values())
    lineas.append("Clases (nº de anotaciones):")
    for clase in sorted(todas | set(clases)):
        lineas.append(f"  {clase}: {clases.get(clase, 0)}")
    lineas.append("Imágenes por carpeta:")
    for carpeta, coco in por_carpeta.items():
        lineas.append(f"  {carpeta}: {len(coco['images'])}")
        nombres += [
            f"{carpeta}/{i['file_name']}"
            for i in sorted(coco["images"], key=lambda i: i["file_name"])
        ]
    paso = max(1, len(nombres) // ejemplos)
    lineas.append(f"Ejemplos de nombres ({min(ejemplos, len(nombres))} de {len(nombres)}):")
    lineas += [f"  {n}" for n in nombres[::paso][:ejemplos]]
    return "\n".join(lineas)


# --- Revisión visual --------------------------------------------------------------------


def revisar(
    salida: str | Path, particion: str, n: int, semilla: int, destino: str | Path
) -> list[Path]:
    """Dibuja cajas y polígonos de ``n`` imágenes al azar de una partición exportada.

    En lenguaje llano: lee el dataset ya exportado (así se comprueba lo que de verdad se ha
    escrito), elige ``n`` imágenes con la semilla dada, pinta cada polígono semitransparente con
    su contorno y su caja, y guarda el resultado en ``<destino>/<partición>/``. Devuelve las
    rutas escritas.
    """
    if particion not in PARTICIONES_EXPORTADAS:
        raise ValueError(
            f"Partición '{particion}' no válida (válidas: {', '.join(PARTICIONES_EXPORTADAS)})"
        )
    carpeta = Path(salida) / particion
    coco = json.loads((carpeta / "_annotations.coco.json").read_text(encoding="utf-8"))
    imagenes = sorted(coco["images"], key=lambda i: i["file_name"])
    elegidas = random.Random(semilla).sample(imagenes, min(n, len(imagenes)))
    anotaciones_de: dict[int, list[dict]] = {}
    for anotacion in coco["annotations"]:
        anotaciones_de.setdefault(anotacion["image_id"], []).append(anotacion)

    escritas = []
    for info in elegidas:
        imagen = leer_imagen(carpeta / info["file_name"])
        capa = imagen.copy()
        poligonos_cv = []
        for anotacion in anotaciones_de.get(info["id"], []):
            for poligono in anotacion["segmentation"]:
                poligonos_cv.append(
                    np.array(poligono, dtype=np.float64).reshape(-1, 2).round().astype(np.int32)
                )
        cv2.fillPoly(capa, poligonos_cv, (0, 200, 0))
        imagen = cv2.addWeighted(capa, 0.3, imagen, 0.7, 0)
        cv2.polylines(imagen, poligonos_cv, True, (0, 200, 0), 1)
        for anotacion in anotaciones_de.get(info["id"], []):
            x, y, ancho, alto = anotacion["bbox"]
            cv2.rectangle(
                imagen, (round(x), round(y)), (round(x + ancho), round(y + alto)), (0, 0, 255), 1
            )
        ruta = Path(destino) / particion / f"{Path(info['file_name']).stem}.jpg"
        ruta.parent.mkdir(parents=True, exist_ok=True)
        guardar_imagen(ruta, imagen)
        escritas.append(ruta)
    return escritas
