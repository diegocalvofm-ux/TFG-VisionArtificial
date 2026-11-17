"""Dataset de recortes para la etapa 2: un recorte por lechuga con sus seis etiquetas.

Qué hace, en lenguaje llano, en este orden:

1. Lee el YAML (``configs/datasets/etapa2_v1.yaml``) y lo valida entero. De cada fuente pública
   toma el ZIP y la licencia de su descriptor (``configs/datasets/<fuente>.yaml``).
2. Junta los *candidatos* (recortes posibles, todavía sin píxeles) de cada fuente:

   - **propio**: una foto ya ingerida (``data/processed/propio_piloto``) por cada fila de
     ``metadata.csv`` de esa fuente y sesión. Las seis etiquetas se copian de la fila.
   - **coco** (YSSP, PHS): una caja por candidato. Su clase se traduce a una etiqueta de un grupo
     (``daño_bio``) y los otros cinco grupos quedan en ``NO_VISIBLE``. Las cajas de *fase* (PHS)
     no son etiqueta: sirven para excluir imágenes (plántulas). Las cajas raras (degeneradas,
     fuera de la imagen o pequeñas) se apartan y se informan.

   Cada candidato recibe su ``recorte_id`` (``<fuente>_<nnnnnn>``) en un orden fijo y ANTES de
   aplicar topes o quitar duplicados: así cambiar un tope no renumera los demás.
3. Si la fuente lo pide, reparte de nuevo por un id sacado del nombre (``reparto``) y aplica el
   tope de recortes sin defecto (``tope_bio_ok``) por partición, con semilla.
4. Recorta: cada imagen se decodifica una sola vez (del ZIP, sin extraerlo, o del disco). En las
   fotos propias, la caja es la de mayor confianza de RF-DETR (o la imagen entera si no detecta
   nada) y se guarda una imagen de control con la caja dibujada. Cada recorte se amplía con el
   margen, se corta en los bordes, se lleva a 512 px de lado mayor y se guarda en JPEG.
5. Busca duplicados con dHash (también girados y en espejo, si se pide). Si un grupo de
   duplicados cruza particiones o fuentes, se queda solo la copia de la partición de más
   prioridad (test > valid > train) y, a igualdad, la de la fuente que va antes en el YAML.
6. Escribe ``recortes.csv`` (validado con las etiquetas de ``schema.py``) e ``informe.json``.

Todo se escribe en ``<salida>.parcial`` y solo al final sustituye a la salida anterior: un fallo
a mitad no deja la salida a medias. Nada de lo escrito lleva rutas absolutas.
"""

import json
import random
import re
import shutil
import zipfile
from collections import Counter, defaultdict
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.config import load_config
from src.data.coco import (
    CARPETAS,
    clases_con_anotaciones,
    leer_anotaciones_zip,
    nombres_de_categorias,
)
from src.data.publicos import PARTICION_SALIDA, leer_metadata
from src.data.recortes import (
    agrupar,
    ampliar_caja,
    huellas,
    motivo_caja_rara,
    pares_cercanos,
    redimensionar_lado_mayor,
    se_sale_en_parte,
)
from src.data.schema import GLOBAL_LEVELS, GROUPS, NO_VISIBLE, VISTAS, is_blank
from src.data.split import _validar_fracciones, group_split
from src.eval.config import DISPOSITIVOS, _numero, _ruta_relativa
from src.eval.metodos import cargar_rfdetr, predictor_rfdetr
from src.eval.salidas import dibujar_cajas
from src.evaluation.aggregate import aggregate
from src.experimentos.privacidad import contiene_ruta_absoluta, mostrar
from src.pipeline.draw import texto_ascii
from src.pipeline.imagenes import decodificar_imagen, guardar_imagen, leer_imagen
from src.pipeline.types import Detection
from src.train.config import _booleano, _entero, _texto

# Particiones del dataset, con los nombres de Roboflow (las carpetas del ZIP).
PARTICIONES = CARPETAS
TIPOS = ("propio", "coco")
FASES = ("incluir", "excluir")
# De dónde sale la caja de cada recorte.
ORIGENES = ("anotacion", "rfdetr", "imagen_entera")
# Motivo con que el informe anota una caja que se usa recortada al borde de la imagen.
MOTIVO_SE_SALE = "se sale de la imagen (se recorta al borde)"

# Columnas de recortes.csv, en orden.
COLUMNAS: tuple[str, ...] = (
    "recorte_id",
    "fuente",
    "licencia",
    "imagen_origen",
    "caja_x",
    "caja_y",
    "caja_w",
    "caja_h",
    "origen_caja",
    "lechuga_id",
    "vista",
    "particion",
    *GROUPS,
    "etiqueta_original",
    "salida_global",
    "dhash",
)

# Etiqueta "sin defecto" de cada grupo: la primera de GROUPS (CUT_OK, ..., BIO_OK). El orden de
# GROUPS lo fija una prueba del esquema. El tope y el reparto estratificado la usan.
SIN_DEFECTO = {grupo: etiquetas[0] for grupo, etiquetas in GROUPS.items()}

# Lo único que puede haber en la carpeta de salida para que se pueda borrar y regenerar.
_CONTENIDO_PROPIO = frozenset(("imagenes", "control", "recortes.csv", "informe.json"))
_NOMBRE_FUENTE = re.compile(r"[A-Za-z0-9_-]+")
_RECORTE_ID = re.compile(r"[A-Za-z0-9_-]+_\d{6}")
_DHASH = re.compile(r"[0-9a-f]{16}")
_EXTENSIONES_PROPIAS = (".jpg", ".jpeg")
_PASO_PROGRESO = 500  # cada cuántas imágenes se avisa del progreso

_CLAVES_RAIZ = ("salida", "semilla", "recorte", "duplicados", "fuentes")
_CLAVES_RECORTE = ("margen", "lado_px", "calidad_jpeg", "min_lado_px")
_CLAVES_DUPLICADOS = ("umbral_hamming", "giros_y_espejos", "prioridad_particiones")
_CLAVES_PROPIO = ("tipo", "imagenes", "metadata", "sesion", "detector", "control_lado_px")
_CLAVES_DETECTOR = ("pesos", "umbral", "dispositivo")
_CLAVES_COCO = ("tipo", "grupo", "clases", "fases", "una_caja_de_estado", "tope_bio_ok", "reparto")
_OBLIGATORIAS_COCO = ("tipo", "grupo", "clases")
_CLAVES_REPARTO = ("grupo_regex", "fracciones")
# Columnas de metadata.csv que hacen falta para las fotos propias.
_COLUMNAS_METADATA = ("image_id", "fuente", "sesion", "licencia", "lechuga_id", "vista",
                      "etiqueta_original", *GROUPS)  # fmt: skip


# --- Configuración -------------------------------------------------------------------------


@dataclass(frozen=True)
class ParametrosRecorte:
    margen: float  # fracción del ancho y del alto de la caja que se añade a cada lado
    lado_px: int  # lado mayor del recorte guardado
    calidad_jpeg: int
    min_lado_px: float  # cajas con el lado menor por debajo: se descartan


@dataclass(frozen=True)
class ParametrosDuplicados:
    umbral_hamming: int
    giros_y_espejos: bool
    prioridad_particiones: tuple[str, ...]  # de más a menos prioridad


@dataclass(frozen=True)
class Detector:
    pesos: str  # relativa a la raíz
    umbral: float  # solo valen las detecciones con confianza MAYOR que esto
    dispositivo: str


@dataclass(frozen=True)
class FuentePropia:
    """Fotos propias ya ingeridas, con sus filas en metadata.csv."""

    nombre: str
    imagenes: Path
    imagenes_rel: str
    metadata: Path
    metadata_rel: str
    sesion: str
    detector: Detector
    control_lado_px: int


@dataclass(frozen=True)
class Reparto:
    """Reparto propio por un id del nombre de archivo, en lugar de las carpetas del ZIP."""

    grupo_regex: re.Pattern
    fracciones: tuple[float, float, float]  # train, valid, test (en número de grupos)


@dataclass(frozen=True)
class FuenteCoco:
    """Un dataset público COCO de Roboflow, leído directamente del ZIP."""

    nombre: str
    licencia: str
    zip: Path
    grupo: str  # grupo de calidad al que se traducen las clases
    clases: dict[str, str]  # clase del ZIP -> etiqueta del grupo
    fases: dict[str, str]  # clase de fase -> incluir | excluir (no son etiqueta)
    una_caja_de_estado: bool  # cada imagen debe tener exactamente una caja de 'clases'
    tope_bio_ok: dict[str, int | None]  # partición -> máximo de recortes sin defecto
    reparto: Reparto | None


@dataclass(frozen=True)
class ConfigEtapa2:
    config_rel: str
    salida: Path
    salida_rel: str
    semilla: int
    recorte: ParametrosRecorte
    duplicados: ParametrosDuplicados
    fuentes: tuple[FuentePropia | FuenteCoco, ...]  # en el orden del YAML (= prioridad)


def _bloque(valor, clave: str) -> dict:
    if not isinstance(valor, dict):
        raise ValueError(f"{clave}: debe ser un bloque con claves (llegó {valor!r})")
    return valor


def _comprobar_claves(bloque: dict, validas, prefijo: str, obligatorias=None) -> None:
    """Error si hay claves desconocidas (casi seguro una errata) o faltan obligatorias."""
    delante = f"{prefijo}." if prefijo else ""
    desconocidas = [f"{delante}{c}" for c in bloque if c not in validas]
    if desconocidas:
        raise ValueError(f"claves desconocidas: {', '.join(map(str, desconocidas))}")
    faltan = [f"{delante}{c}" for c in (validas if obligatorias is None else obligatorias)
              if c not in bloque]  # fmt: skip
    if faltan:
        raise ValueError(f"faltan claves: {', '.join(faltan)}")


def _tope(valor, clave: str) -> dict[str, int | None]:
    """``null`` (sin tope), un entero (el mismo en cada partición) o ``{partición: entero}``."""
    if valor is None:
        return dict.fromkeys(PARTICIONES)
    if isinstance(valor, dict):
        sobran = [p for p in valor if p not in PARTICIONES]
        if sobran:
            raise ValueError(f"{clave}: particiones desconocidas {sobran} (válidas: "
                             f"{', '.join(PARTICIONES)})")  # fmt: skip
        return {p: None if valor.get(p) is None else _entero(valor[p], f"{clave}.{p}", 0)
                for p in PARTICIONES}  # fmt: skip
    numero = _entero(valor, clave, 0)
    return dict.fromkeys(PARTICIONES, numero)


def _reparto(valor, clave: str) -> Reparto:
    bloque = _bloque(valor, clave)
    _comprobar_claves(bloque, _CLAVES_REPARTO, clave)
    try:
        regex = re.compile(_texto(bloque["grupo_regex"], f"{clave}.grupo_regex"))
    except re.error as error:
        raise ValueError(f"{clave}.grupo_regex no válida: {error}") from error
    crudas = _bloque(bloque["fracciones"], f"{clave}.fracciones")
    if set(crudas) != set(PARTICIONES):
        raise ValueError(f"{clave}.fracciones: necesita las claves {', '.join(PARTICIONES)}")
    fracciones = tuple(crudas[p] for p in PARTICIONES)
    if any(isinstance(f, bool) or not isinstance(f, int | float) for f in fracciones):
        raise ValueError(f"{clave}.fracciones: deben ser números (llegó {fracciones})")
    try:
        _validar_fracciones(fracciones)
    except ValueError as error:
        raise ValueError(f"{clave}.fracciones: {error}") from error
    return Reparto(grupo_regex=regex, fracciones=tuple(float(f) for f in fracciones))


def _cargar_propia(nombre: str, bloque: dict, raiz: Path) -> FuentePropia:
    prefijo = f"fuentes.{nombre}"
    _comprobar_claves(bloque, _CLAVES_PROPIO, prefijo)
    imagenes = _ruta_relativa(bloque["imagenes"], f"{prefijo}.imagenes")
    metadata = _ruta_relativa(bloque["metadata"], f"{prefijo}.metadata")
    detector = _bloque(bloque["detector"], f"{prefijo}.detector")
    _comprobar_claves(detector, _CLAVES_DETECTOR, f"{prefijo}.detector")
    fuente = FuentePropia(
        nombre=nombre,
        imagenes=raiz / imagenes,
        imagenes_rel=imagenes,
        metadata=raiz / metadata,
        metadata_rel=metadata,
        sesion=_texto(bloque["sesion"], f"{prefijo}.sesion"),
        detector=Detector(
            pesos=_ruta_relativa(detector["pesos"], f"{prefijo}.detector.pesos"),
            umbral=float(_numero(detector["umbral"], f"{prefijo}.detector.umbral", 0, 1)),
            dispositivo=_opcion_dispositivo(detector["dispositivo"], prefijo),
        ),
        control_lado_px=_entero(bloque["control_lado_px"], f"{prefijo}.control_lado_px", 1),
    )
    if not fuente.imagenes.is_dir():
        raise FileNotFoundError(f"Fuente '{nombre}': no existe la carpeta de fotos {imagenes}")
    if not fuente.metadata.is_file():
        raise FileNotFoundError(f"Fuente '{nombre}': no existe {metadata}")
    return fuente


def _opcion_dispositivo(valor, prefijo: str) -> str:
    if valor not in DISPOSITIVOS:
        raise ValueError(f"{prefijo}.detector.dispositivo: debe ser uno de "
                         f"{', '.join(DISPOSITIVOS)} (llegó {valor!r})")  # fmt: skip
    return valor


def _cargar_coco(nombre: str, bloque: dict, carpeta: Path, raiz: Path) -> FuenteCoco:
    """Bloque de una fuente COCO del YAML más su descriptor ``<carpeta>/<nombre>.yaml``."""
    prefijo = f"fuentes.{nombre}"
    _comprobar_claves(bloque, _CLAVES_COCO, prefijo, _OBLIGATORIAS_COCO)
    grupo = bloque["grupo"]
    if grupo not in GROUPS:
        raise ValueError(f"{prefijo}.grupo: '{grupo}' no existe (válidos: {', '.join(GROUPS)})")
    clases = _bloque(bloque["clases"], f"{prefijo}.clases")
    if not clases:
        raise ValueError(f"{prefijo}.clases: no puede estar vacío")
    for clase, etiqueta in clases.items():
        if etiqueta not in GROUPS[grupo]:
            raise ValueError(
                f"{prefijo}.clases.{clase}: '{etiqueta}' no es una etiqueta de '{grupo}' "
                f"(válidas: {', '.join(GROUPS[grupo])})"
            )
    fases = _bloque(bloque.get("fases") or {}, f"{prefijo}.fases")
    for clase, accion in fases.items():
        if accion not in FASES:
            raise ValueError(f"{prefijo}.fases.{clase}: debe ser {' o '.join(FASES)} "
                             f"(llegó {accion!r})")  # fmt: skip
        if clase in clases:
            raise ValueError(f"{prefijo}: la clase '{clase}' está a la vez en clases y en fases")
    reparto = bloque.get("reparto")

    descriptor = carpeta / f"{nombre}.yaml"
    if not descriptor.is_file():
        raise FileNotFoundError(
            f"Fuente '{nombre}': no se encuentra su descriptor {mostrar(descriptor, raiz)} "
            "(con fuente, licencia y zip)"
        )
    datos = load_config(descriptor)
    if datos.get("fuente") != nombre:
        raise ValueError(f"{descriptor.name}: 'fuente' debe ser '{nombre}'")
    licencia = _texto(datos.get("licencia"), f"{descriptor.name}: licencia")
    zip_rel = _ruta_relativa(datos.get("zip"), f"{descriptor.name}: zip")
    if not (raiz / zip_rel).is_file():
        raise FileNotFoundError(f"Fuente '{nombre}': no se encuentra el ZIP {zip_rel}")
    return FuenteCoco(
        nombre=nombre,
        licencia=licencia.strip(),
        zip=raiz / zip_rel,
        grupo=grupo,
        clases={str(k): v for k, v in clases.items()},
        fases={str(k): v for k, v in fases.items()},
        una_caja_de_estado=_booleano(bloque.get("una_caja_de_estado", False),
                                     f"{prefijo}.una_caja_de_estado"),  # fmt: skip
        tope_bio_ok=_tope(bloque.get("tope_bio_ok"), f"{prefijo}.tope_bio_ok"),
        reparto=None if reparto is None else _reparto(reparto, f"{prefijo}.reparto"),
    )


def _validar(config: dict, ruta: Path, raiz: Path) -> ConfigEtapa2:
    _comprobar_claves(config, _CLAVES_RAIZ, "")
    salida = _ruta_relativa(config["salida"], "salida")
    recorte = _bloque(config["recorte"], "recorte")
    _comprobar_claves(recorte, _CLAVES_RECORTE, "recorte")
    calidad = _entero(recorte["calidad_jpeg"], "recorte.calidad_jpeg", 1)
    if calidad > 100:
        raise ValueError(f"recorte.calidad_jpeg: debe estar entre 1 y 100 (llegó {calidad})")
    duplicados = _bloque(config["duplicados"], "duplicados")
    _comprobar_claves(duplicados, _CLAVES_DUPLICADOS, "duplicados")
    umbral = _entero(duplicados["umbral_hamming"], "duplicados.umbral_hamming", 0)
    if umbral > 64:
        raise ValueError(f"duplicados.umbral_hamming: una huella tiene 64 bits (llegó {umbral})")
    prioridad = duplicados["prioridad_particiones"]
    if not isinstance(prioridad, list) or sorted(prioridad) != sorted(PARTICIONES):
        raise ValueError(
            "duplicados.prioridad_particiones: debe ordenar train, valid y test, cada una una "
            f"vez (llegó {prioridad!r})"
        )

    crudas = _bloque(config["fuentes"], "fuentes")
    if not crudas:
        raise ValueError("fuentes: hace falta al menos una fuente")
    fuentes = []
    for nombre, bloque in crudas.items():
        if not isinstance(nombre, str) or not _NOMBRE_FUENTE.fullmatch(nombre):
            raise ValueError(f"fuentes: '{nombre}' no es un nombre válido (letras, números, - y _)")
        bloque = _bloque(bloque, f"fuentes.{nombre}")
        tipo = bloque.get("tipo")
        if tipo == "propio":
            fuentes.append(_cargar_propia(nombre, bloque, raiz))
        elif tipo == "coco":
            fuentes.append(_cargar_coco(nombre, bloque, ruta.parent, raiz))
        else:
            raise ValueError(f"fuentes.{nombre}.tipo: debe ser {' o '.join(TIPOS)} "
                             f"(llegó {tipo!r})")  # fmt: skip

    return ConfigEtapa2(
        config_rel=mostrar(ruta, raiz),
        salida=raiz / salida,
        salida_rel=salida,
        semilla=_entero(config["semilla"], "semilla", 0),
        recorte=ParametrosRecorte(
            margen=float(_numero(recorte["margen"], "recorte.margen", 0, 1)),
            lado_px=_entero(recorte["lado_px"], "recorte.lado_px", 1),
            calidad_jpeg=calidad,
            min_lado_px=float(_numero(recorte["min_lado_px"], "recorte.min_lado_px", 0, 1e6)),
        ),
        duplicados=ParametrosDuplicados(
            umbral_hamming=umbral,
            giros_y_espejos=_booleano(duplicados["giros_y_espejos"], "duplicados.giros_y_espejos"),
            prioridad_particiones=tuple(prioridad),
        ),
        fuentes=tuple(fuentes),
    )


def cargar_config(ruta: str | Path, raiz: str | Path) -> ConfigEtapa2:
    """Lee y valida el YAML del dataset de la etapa 2 (y el descriptor de cada fuente COCO).

    En lenguaje llano: comprueba que no falta ni sobra ninguna clave (una errata no pasa), que los
    números están en su rango, que las rutas son relativas a la raíz y existen, y que cada clase
    se traduce a una etiqueta de su grupo según ``schema.py``. El ZIP y la licencia de una fuente
    pública salen de ``configs/datasets/<fuente>.yaml`` (junto al YAML de la etapa).

    Errores: ``ValueError`` (con el nombre del YAML y la clave) si algo no cuadra;
    ``FileNotFoundError`` (con la ruta relativa) si falta un archivo o una carpeta.
    """
    ruta, raiz = Path(ruta), Path(raiz)
    config = load_config(ruta)
    try:
        return _validar(config, ruta, raiz)
    except ValueError as error:
        raise ValueError(f"{ruta.name}: {error}") from error


# --- Candidatos ----------------------------------------------------------------------------


@dataclass
class Candidato:
    """Un recorte posible: de qué imagen sale, su caja y sus etiquetas (aún sin píxeles)."""

    recorte_id: str
    fuente: str
    licencia: str
    imagen_origen: str  # nombre del archivo, sin carpetas
    ruta: str  # dentro del ZIP ("train/a.jpg") o nombre en la carpeta de fotos propias
    lechuga_id: str
    vista: str
    particion: str
    etiquetas: dict[str, str]
    etiqueta_original: str
    caja: tuple[float, float, float, float] | None = None  # (x, y, w, h); propio: tras detectar
    origen_caja: str = "anotacion"
    ancho: int | None = None  # tamaño que dice el JSON (solo COCO)
    alto: int | None = None
    huellas: tuple[int, ...] = ()


@dataclass
class Lectura:
    """Lo que sale de leer una fuente: candidatos y lo apartado (para el informe)."""

    candidatos: list[Candidato] = field(default_factory=list)
    plantulas: list[str] = field(default_factory=list)
    cajas_raras: list[dict] = field(default_factory=list)
    avisos: list[str] = field(default_factory=list)


def _caja_rara(fuente: str, carpeta: str, nombre: str, motivo: str, *, descartada: bool,
               recorte_id: str | None = None, caja=None) -> dict:  # fmt: skip
    return {
        "fuente": fuente,
        "particion_zip": carpeta,
        "imagen_origen": nombre,
        "recorte_id": recorte_id,
        "motivo": motivo,
        "descartada": descartada,
        "caja": None if caja is None else [float(v) for v in caja],
    }


def candidatos_coco(fuente: FuenteCoco, min_lado_px: float) -> Lectura:
    """Candidatos de un ZIP COCO: una caja de estado = un recorte.

    En lenguaje llano:

    - Se cuentan las anotaciones de cada clase por su nombre. Una clase con anotaciones que no
      esté ni en ``clases`` ni en ``fases`` es un error (no se sabría qué etiqueta darle). Las
      categorías sin anotaciones (la "padre" de Roboflow) no cuentan.
    - Se recorren las carpetas train, valid y test y, en cada una, las imágenes por nombre. Cada
      caja de estado (clase de ``clases``) gasta un número de ``recorte_id``, también las que
      luego se apartan: así los ids no dependen de lo apartado.
    - Una imagen con una caja de fase ``excluir`` (plántula) sale entera del dataset. Con
      ``una_caja_de_estado``, una imagen sin exactamente una caja de estado se salta.
    - Las cajas degeneradas, fuera de la imagen o pequeñas se apartan; las que se salen solo en
      parte se usan recortadas al borde. Todo queda anotado en ``cajas_raras``.

    La partición es la carpeta del ZIP y el ``lechuga_id`` la imagen de origen (sus recortes no
    se separan). El ZIP solo se lee.
    """
    with zipfile.ZipFile(fuente.zip) as zf:
        por_carpeta = leer_anotaciones_zip(zf)
        contenido = set(zf.namelist())

    totales: Counter = Counter()
    for coco in por_carpeta.values():
        totales.update(clases_con_anotaciones(coco))
    sin_mapeo = sorted(c for c in totales if c not in fuente.clases and c not in fuente.fases)
    if sin_mapeo:
        detalle = ", ".join(f"'{c}' ({totales[c]} anotaciones)" for c in sin_mapeo)
        raise ValueError(
            f"Fuente '{fuente.nombre}': clase(s) del ZIP con anotaciones y sin mapeo en el YAML: "
            f"{detalle}. Añádelas a 'clases' (con su etiqueta) o a 'fases' (incluir o excluir)"
        )
    lectura = Lectura()
    for clase in (*fuente.clases, *fuente.fases):
        if clase not in totales:
            lectura.avisos.append(
                f"Fuente '{fuente.nombre}': la clase '{clase}' del YAML no tiene anotaciones en "
                "el ZIP"
            )

    numero = 0
    for carpeta in CARPETAS:
        coco = por_carpeta.get(carpeta)
        if coco is None:
            continue
        categorias = nombres_de_categorias(coco)
        anotaciones_de: dict[int, list[dict]] = defaultdict(list)
        for anotacion in coco["annotations"]:
            anotaciones_de[anotacion["image_id"]].append(anotacion)

        for info in sorted(coco["images"], key=lambda i: i["file_name"]):
            nombre = info["file_name"]
            ruta = f"{carpeta}/{nombre}"
            if ruta not in contenido:
                raise ValueError(
                    f"Fuente '{fuente.nombre}': el JSON cita '{ruta}' pero no está en el ZIP"
                )
            anotaciones = sorted(anotaciones_de.get(info["id"], []),
                                 key=lambda a: (tuple(a["bbox"]), a["id"]))  # fmt: skip
            estado = [a for a in anotaciones if categorias[a["category_id"]] in fuente.clases]
            fases = {categorias[a["category_id"]] for a in anotaciones
                     if categorias[a["category_id"]] in fuente.fases}  # fmt: skip
            ids = []
            for _ in estado:
                numero += 1
                ids.append(f"{fuente.nombre}_{numero:06d}")

            if any(fuente.fases[f] == "excluir" for f in fases):
                lectura.plantulas.append(nombre)
                continue
            if fuente.una_caja_de_estado and len(estado) != 1:
                lectura.cajas_raras.append(_caja_rara(
                    fuente.nombre, carpeta, nombre,
                    f"{len(estado)} cajas de estado (se esperaba 1): se salta la imagen",
                    descartada=True,
                ))  # fmt: skip
                continue

            ancho, alto = int(info["width"]), int(info["height"])
            for anotacion, recorte_id in zip(estado, ids, strict=True):
                caja = tuple(float(v) for v in anotacion["bbox"])
                motivo = motivo_caja_rara(caja, ancho, alto, min_lado_px)
                if motivo is not None:
                    lectura.cajas_raras.append(_caja_rara(
                        fuente.nombre, carpeta, nombre, motivo, descartada=True,
                        recorte_id=recorte_id, caja=caja,
                    ))  # fmt: skip
                    continue
                if se_sale_en_parte(caja, ancho, alto):
                    lectura.cajas_raras.append(_caja_rara(
                        fuente.nombre, carpeta, nombre, MOTIVO_SE_SALE,
                        descartada=False, recorte_id=recorte_id, caja=caja,
                    ))  # fmt: skip
                clase = categorias[anotacion["category_id"]]
                etiquetas = dict.fromkeys(GROUPS, NO_VISIBLE)
                etiquetas[fuente.grupo] = fuente.clases[clase]
                lectura.candidatos.append(
                    Candidato(
                        recorte_id=recorte_id,
                        fuente=fuente.nombre,
                        licencia=fuente.licencia,
                        imagen_origen=nombre,
                        ruta=ruta,
                        lechuga_id=f"{fuente.nombre}_{Path(nombre).stem}",
                        vista="",
                        particion=carpeta,
                        etiquetas=etiquetas,
                        etiqueta_original=clase,
                        caja=caja,
                        origen_caja="anotacion",
                        ancho=ancho,
                        alto=alto,
                    )
                )
    return lectura


def candidatos_propio(fuente: FuentePropia) -> Lectura:
    """Candidatos de las fotos propias: uno por foto, con las etiquetas de metadata.csv.

    En lenguaje llano: las filas de ``metadata.csv`` de esta fuente y sesión tienen que casar una
    a una con los jpg de la carpeta (por ``image_id`` = nombre sin extensión). Una foto sin fila o
    una fila sin foto es un error; se juntan todos en un único mensaje. La caja aún no se sabe:
    la pone RF-DETR al recortar. Todas van a train.
    """
    metadata = leer_metadata(fuente.metadata)
    if metadata is None:
        raise FileNotFoundError(f"Fuente '{fuente.nombre}': no existe {fuente.metadata_rel}")
    faltan = [c for c in _COLUMNAS_METADATA if c not in metadata.columns]
    if faltan:
        raise ValueError(f"Fuente '{fuente.nombre}': a {fuente.metadata_rel} le faltan las "
                         f"columnas {', '.join(faltan)}")  # fmt: skip
    filas = metadata[(metadata["fuente"] == fuente.nombre) & (metadata["sesion"] == fuente.sesion)]

    errores = []
    por_id: dict[str, dict] = {}
    for fila in filas.to_dict("records"):
        if fila["image_id"] in por_id:
            errores.append(f"image_id '{fila['image_id']}' repetido en {fuente.metadata_rel}")
        por_id[fila["image_id"]] = fila
    fotos: dict[str, str] = {}
    for ruta in sorted(fuente.imagenes.iterdir(), key=lambda p: p.name):
        if not ruta.is_file() or ruta.suffix.lower() not in _EXTENSIONES_PROPIAS:
            continue
        if ruta.stem in fotos:
            errores.append(f"dos fotos con el mismo image_id: {fotos[ruta.stem]} y {ruta.name}")
        fotos[ruta.stem] = ruta.name

    donde = f"(fuente {fuente.nombre}, sesión {fuente.sesion})"
    sin_fila = sorted(set(fotos) - set(por_id))
    sin_foto = sorted(set(por_id) - set(fotos))
    if sin_fila:
        errores.append(f"{len(sin_fila)} foto(s) de {fuente.imagenes_rel} sin fila en "
                       f"{fuente.metadata_rel} {donde}: {', '.join(sin_fila)}")  # fmt: skip
    if sin_foto:
        errores.append(f"{len(sin_foto)} fila(s) de {fuente.metadata_rel} {donde} sin foto en "
                       f"{fuente.imagenes_rel}: {', '.join(sin_foto)}")  # fmt: skip
    if not fotos and not errores:
        errores.append(f"no hay fotos en {fuente.imagenes_rel}")
    if errores:
        detalle = "\n  - ".join(errores)
        raise ValueError(f"Fuente '{fuente.nombre}': {len(errores)} error(es):\n  - {detalle}")

    lectura = Lectura()
    for numero, image_id in enumerate(sorted(fotos), start=1):
        fila = por_id[image_id]
        lectura.candidatos.append(
            Candidato(
                recorte_id=f"{fuente.nombre}_{numero:06d}",
                fuente=fuente.nombre,
                licencia=fila["licencia"],
                imagen_origen=fotos[image_id],
                ruta=fotos[image_id],
                lechuga_id=fila["lechuga_id"],
                vista=fila["vista"],
                particion="train",
                etiquetas={grupo: fila[grupo] for grupo in GROUPS},
                etiqueta_original=fila["etiqueta_original"],
                origen_caja="",
            )
        )
    return lectura


def repartir_por_grupo(candidatos: list[Candidato], fuente: FuenteCoco, semilla: int) -> dict:
    """Reparte los candidatos por el id que ``reparto.grupo_regex`` saca del nombre del archivo.

    En lenguaje llano: todas las imágenes con el mismo id (en PHS, la misma cama o cámara) van a
    la misma partición, así train y test no comparten id. Para que cada partición tenga de los dos
    tipos, los ids "con defecto" (alguna caja con una etiqueta distinta de la de sin defecto) y
    los "sanos" se reparten por separado con ``group_split`` (misma semilla y fracciones, que se
    cumplen en número de ids). El ``lechuga_id`` pasa a ser ``<fuente>_<id>``. Modifica los
    candidatos y devuelve el resumen para el informe.
    """
    reparto = fuente.reparto
    ids: dict[str, str] = {}
    for candidato in candidatos:
        coincidencia = reparto.grupo_regex.search(candidato.imagen_origen)
        if coincidencia is None:
            raise ValueError(
                f"Fuente '{fuente.nombre}': el nombre '{candidato.imagen_origen}' no encaja con "
                f"reparto.grupo_regex '{reparto.grupo_regex.pattern}'"
            )
        ids[candidato.recorte_id] = (
            coincidencia.group(1) if reparto.grupo_regex.groups else coincidencia.group(0)
        )
    sin_defecto = SIN_DEFECTO[fuente.grupo]
    con_defecto = sorted({ids[c.recorte_id] for c in candidatos
                          if c.etiquetas[fuente.grupo] != sin_defecto})  # fmt: skip
    sanos = sorted(set(ids.values()) - set(con_defecto))

    asignacion: dict[str, str] = {}
    for tipo, grupos in (("con defecto", con_defecto), ("sanos", sanos)):
        if not grupos:
            continue
        try:
            resultado = group_split(pd.DataFrame({"grupo": grupos}), "grupo",
                                    reparto.fracciones, semilla)  # fmt: skip
        except ValueError as error:
            raise ValueError(
                f"Fuente '{fuente.nombre}': reparto de los ids {tipo} ({len(grupos)}): {error}"
            ) from error
        for grupo, particion in zip(resultado["grupo"], resultado["split"], strict=True):
            asignacion[grupo] = PARTICION_SALIDA[particion]

    for candidato in candidatos:
        grupo = ids[candidato.recorte_id]
        candidato.particion = asignacion[grupo]
        candidato.lechuga_id = f"{fuente.nombre}_{grupo}"
    resumen = {}
    for particion in PARTICIONES:
        grupos = sorted(g for g, p in asignacion.items() if p == particion)
        de_aqui = [c for c in candidatos if c.particion == particion]
        resumen[particion] = {
            "grupos": grupos,
            "grupos_con_defecto": [g for g in grupos if g in con_defecto],
            "imagenes": len({c.imagen_origen for c in de_aqui}),
            "recortes": len(de_aqui),
        }
    return resumen


def aplicar_tope_bio_ok(
    candidatos: list[Candidato], fuente: FuenteCoco, semilla: int
) -> tuple[list[Candidato], dict]:
    """Deja como mucho ``tope`` candidatos sin defecto (BIO_OK) por partición, elegidos al azar.

    En lenguaje llano: en cada partición se toman los candidatos cuya etiqueta del grupo es la de
    sin defecto, en su orden fijo, y si pasan del tope se eligen ``tope`` con
    ``random.Random(semilla)``: misma semilla, misma selección. Los que tienen un defecto no se
    tocan. Devuelve (candidatos que quedan, resumen por partición).
    """
    sin_defecto = SIN_DEFECTO[fuente.grupo]
    quitar: set[str] = set()
    resumen = {}
    for particion in PARTICIONES:
        sanos = [
            c.recorte_id
            for c in candidatos
            if c.particion == particion and c.etiquetas[fuente.grupo] == sin_defecto
        ]
        tope = fuente.tope_bio_ok[particion]
        elegidos = sanos
        if tope is not None and len(sanos) > tope:
            elegidos = random.Random(semilla).sample(sanos, tope)
        quitar |= set(sanos) - set(elegidos)
        resumen[particion] = {"tope": tope, "candidatos": len(sanos), "elegidos": len(elegidos)}
    return [c for c in candidatos if c.recorte_id not in quitar], resumen


@dataclass
class Seleccion:
    """Candidatos de todas las fuentes y lo que el informe cuenta de su selección."""

    candidatos: list[Candidato] = field(default_factory=list)
    plantulas: dict[str, dict] = field(default_factory=dict)
    cajas_raras: list[dict] = field(default_factory=list)
    reparto: dict[str, dict] = field(default_factory=dict)
    tope: dict[str, dict] = field(default_factory=dict)
    avisos: list[str] = field(default_factory=list)


def seleccionar(config: ConfigEtapa2, progreso: Callable[[str], None] | None = None) -> Seleccion:
    """Lee las fuentes en el orden del YAML y aplica reparto y tope (sin tocar píxeles)."""
    seleccion = Seleccion()
    for fuente in config.fuentes:
        if progreso:
            que = "metadata.csv" if isinstance(fuente, FuentePropia) else "anotaciones del ZIP"
            progreso(f"{fuente.nombre}: leyendo {que}")
        if isinstance(fuente, FuentePropia):
            lectura = candidatos_propio(fuente)
            seleccion.candidatos += lectura.candidatos
            continue
        lectura = candidatos_coco(fuente, config.recorte.min_lado_px)
        candidatos = lectura.candidatos
        if fuente.reparto is not None:
            seleccion.reparto[fuente.nombre] = repartir_por_grupo(candidatos, fuente,
                                                                  config.semilla)  # fmt: skip
        candidatos, seleccion.tope[fuente.nombre] = aplicar_tope_bio_ok(
            candidatos, fuente, config.semilla
        )
        seleccion.candidatos += candidatos
        if fuente.fases:
            seleccion.plantulas[fuente.nombre] = {"imagenes": len(lectura.plantulas),
                                                  "nombres": lectura.plantulas}  # fmt: skip
        seleccion.cajas_raras += lectura.cajas_raras
        seleccion.avisos += lectura.avisos
    return seleccion


# --- Recortes ------------------------------------------------------------------------------


def _guardar_control(
    ruta: Path, imagen: np.ndarray, recorte: tuple, deteccion: Detection | None, lado: int
) -> None:
    """Imagen de control de una foto propia: reducida, con la caja de recorte y la detección.

    En verde la caja que se recorta (con su margen) y en rojo la de RF-DETR con su confianza
    (reutiliza ``dibujar_cajas`` de la evaluación). Sin detección, un rótulo lo dice.
    """
    alto, ancho = imagen.shape[:2]
    escala = min(1.0, lado / max(alto, ancho))
    pequena = imagen
    if escala < 1:
        nuevo = (max(1, round(ancho * escala)), max(1, round(alto * escala)))
        pequena = cv2.resize(imagen, nuevo, interpolation=cv2.INTER_AREA)

    def escalar(caja):
        return tuple(round(v * escala) for v in caja)

    predicciones = []
    if deteccion is not None:
        predicciones = [Detection(box=escalar(deteccion.box), score=deteccion.score)]
    dibujada = dibujar_cajas(pequena, [escalar(recorte)], predicciones)
    if deteccion is None:
        tamano = max(0.3, max(dibujada.shape[:2]) / 900)
        alto_texto = cv2.getTextSize("X", cv2.FONT_HERSHEY_SIMPLEX, tamano, 1)[0][1]
        cv2.putText(dibujada, texto_ascii("SIN DETECCION"), (3, alto_texto + 3),
                    cv2.FONT_HERSHEY_SIMPLEX, tamano, (0, 0, 255), 1)  # fmt: skip
    guardar_imagen(ruta, dibujada)


def _detectar(
    candidato: Candidato,
    imagen: np.ndarray,
    fuente: FuentePropia,
    predecir: Callable[[np.ndarray], list[Detection]],
    margen: float,
    carpeta_control: Path,
    resumen: dict,
    avisos: list[str],
) -> None:
    """Pone la caja de una foto propia: la de mayor confianza de RF-DETR o la imagen entera.

    En lenguaje llano: solo valen las detecciones con confianza MAYOR que el umbral y con área;
    llegan ordenadas de mayor a menor, así que se toma la primera (recortada a la imagen). Si no
    hay ninguna, se usa la imagen entera y se avisa. Guarda la imagen de control.
    """
    alto, ancho = imagen.shape[:2]
    image_id = Path(candidato.ruta).stem
    validas = []
    for deteccion in predecir(imagen):
        x1, y1, x2, y2 = deteccion.box
        x1, x2 = max(0, x1), min(ancho, x2)
        y1, y2 = max(0, y1), min(alto, y2)
        if deteccion.score > fuente.detector.umbral and x2 > x1 and y2 > y1:
            validas.append(Detection(box=(x1, y1, x2, y2), score=deteccion.score))
    resumen["fotos"] += 1
    mejor = validas[0] if validas else None
    if mejor is not None:
        x1, y1, x2, y2 = mejor.box
        candidato.caja = (x1, y1, x2 - x1, y2 - y1)
        candidato.origen_caja = "rfdetr"
        resumen["detecciones"].append({"image_id": image_id, "confianza": round(mejor.score, 4)})
    else:
        candidato.caja = (0, 0, ancho, alto)
        candidato.origen_caja = "imagen_entera"
        resumen["sin_deteccion"].append(image_id)
        avisos.append(
            f"Fuente '{fuente.nombre}': {image_id} sin detección de RF-DETR por encima de "
            f"{fuente.detector.umbral}: se usa la imagen entera"
        )
    recorte = ampliar_caja(candidato.caja, margen, ancho, alto)
    _guardar_control(carpeta_control / f"{image_id}.jpg", imagen, recorte, mejor,
                     fuente.control_lado_px)  # fmt: skip


def materializar(
    candidatos: list[Candidato],
    config: ConfigEtapa2,
    carpeta: Path,
    predictores: dict[str, Callable[[np.ndarray], list[Detection]]],
    progreso: Callable[[str], None] | None = None,
) -> tuple[list[Candidato], dict, list[dict], list[str]]:
    """Recorta, huella y guarda cada candidato en ``<carpeta>/imagenes/<recorte_id>.jpg``.

    En lenguaje llano: se agrupan los candidatos por imagen para decodificarla una sola vez (del
    ZIP con ``read``, sin extraer nada, o de la carpeta de fotos propias). En las propias se
    detecta antes la caja. Cada recorte: caja + margen, corte en los bordes, lado mayor a
    ``lado_px`` sin deformar, huellas (con giros y espejos si se pide) y JPEG con la calidad del
    YAML (OpenCV no escribe metadatos). Si el tamaño real de una imagen no coincide con el del
    JSON (las cajas no valdrían) sus recortes se apartan como cajas raras.

    Devuelve (candidatos recortados, resumen de las fotos propias, cajas raras nuevas, avisos).
    """
    fuentes = {f.nombre: f for f in config.fuentes}
    por_imagen: dict[tuple[str, str], list[Candidato]] = {}
    for candidato in candidatos:
        por_imagen.setdefault((candidato.fuente, candidato.ruta), []).append(candidato)
    totales = Counter(nombre for nombre, _ in por_imagen)
    hechas: Counter = Counter()
    resumen_propio = {f.nombre: {"fotos": 0, "sin_deteccion": [], "detecciones": []}
                      for f in config.fuentes if isinstance(f, FuentePropia)}  # fmt: skip
    hechos, raras, avisos = [], [], []

    with ExitStack() as pila:
        zips: dict[str, zipfile.ZipFile] = {}
        for (nombre, ruta), grupo in por_imagen.items():
            fuente = fuentes[nombre]
            if isinstance(fuente, FuenteCoco):
                if nombre not in zips:
                    zips[nombre] = pila.enter_context(zipfile.ZipFile(fuente.zip))
                imagen = decodificar_imagen(zips[nombre].read(ruta), ruta)
            else:
                imagen = leer_imagen(fuente.imagenes / ruta)
                _detectar(grupo[0], imagen, fuente, predictores[nombre], config.recorte.margen,
                          carpeta / "control", resumen_propio[nombre], avisos)  # fmt: skip
            alto, ancho = imagen.shape[:2]
            esperado = (grupo[0].ancho, grupo[0].alto)
            if isinstance(fuente, FuenteCoco) and esperado != (ancho, alto):
                for candidato in grupo:
                    raras.append(_caja_rara(
                        nombre, ruta.split("/")[0], candidato.imagen_origen,
                        f"la imagen mide {ancho}x{alto} y el JSON dice "
                        f"{esperado[0]}x{esperado[1]}: se aparta", descartada=True,
                        recorte_id=candidato.recorte_id, caja=candidato.caja,
                    ))  # fmt: skip
                continue
            for candidato in grupo:
                x1, y1, x2, y2 = ampliar_caja(candidato.caja, config.recorte.margen, ancho, alto)
                recorte = redimensionar_lado_mayor(imagen[y1:y2, x1:x2], config.recorte.lado_px)
                candidato.huellas = huellas(recorte, config.duplicados.giros_y_espejos)
                guardar_imagen(carpeta / "imagenes" / f"{candidato.recorte_id}.jpg", recorte,
                               calidad_jpeg=config.recorte.calidad_jpeg)  # fmt: skip
                hechos.append(candidato)
            hechas[nombre] += 1
            if progreso and (hechas[nombre] % _PASO_PROGRESO == 0
                             or hechas[nombre] == totales[nombre]):  # fmt: skip
                progreso(f"{nombre}: {hechas[nombre]}/{totales[nombre]} imágenes recortadas")
    return hechos, resumen_propio, raras, avisos


# --- Duplicados ----------------------------------------------------------------------------


def resolver_duplicados(
    recortes: list[Candidato], parametros: ParametrosDuplicados, orden_fuentes: list[str]
) -> tuple[set[str], dict]:
    """Busca duplicados por dHash y decide cuáles se quitan. Devuelve (ids a quitar, resumen).

    En lenguaje llano: dos recortes son duplicados si sus huellas están a distancia de Hamming
    menor o igual que el umbral (contando giros y espejos si se calcularon); las cadenas de
    duplicados forman grupos. Una *celda* es una (fuente, partición). Si un grupo abarca más de
    una celda, se queda entera la celda con más prioridad (primero la partición según
    ``prioridad_particiones``; a igualdad, la fuente que va antes en el YAML) y se quitan los
    recortes de las demás. Los grupos dentro de una sola celda no se tocan: solo se informan.
    """
    pares = pares_cercanos([c.huellas for c in recortes], parametros.umbral_hamming)
    grupos = agrupar(len(recortes), pares)
    grupo_de = {i: n for n, grupo in enumerate(grupos) for i in grupo}
    distancia_max: dict[int, int] = defaultdict(int)
    for i, _, d in pares:
        distancia_max[grupo_de[i]] = max(distancia_max[grupo_de[i]], d)

    def prioridad(celda: tuple[str, str]) -> tuple[int, int]:
        fuente, particion = celda
        return parametros.prioridad_particiones.index(particion), orden_fuentes.index(fuente)

    quitar: set[str] = set()
    cruzan, dentro = [], []
    for n, grupo in enumerate(grupos):
        miembros = [recortes[i] for i in grupo]
        celdas = {(c.fuente, c.particion) for c in miembros}
        if len(celdas) == 1:
            fuente, particion = next(iter(celdas))
            dentro.append({"fuente": fuente, "particion": particion,
                           "distancia_max": distancia_max[n],
                           "recortes": [c.recorte_id for c in miembros]})  # fmt: skip
            continue
        queda = min(celdas, key=prioridad)
        detalle = []
        for c in miembros:
            se_quita = (c.fuente, c.particion) != queda
            if se_quita:
                quitar.add(c.recorte_id)
            detalle.append({"recorte_id": c.recorte_id, "fuente": c.fuente,
                            "particion": c.particion, "imagen_origen": c.imagen_origen,
                            "se_quita": se_quita})  # fmt: skip
        cruzan.append({"se_queda": {"fuente": queda[0], "particion": queda[1]},
                       "distancia_max": distancia_max[n], "recortes": detalle})  # fmt: skip
    resumen = {
        "umbral_hamming": parametros.umbral_hamming,
        "giros_y_espejos": parametros.giros_y_espejos,
        "prioridad_particiones": list(parametros.prioridad_particiones),
        "grupos_que_cruzan": len(cruzan),
        "recortes_quitados": len(quitar),
        "grupos": cruzan,
        "dentro_de_una_celda": {
            "grupos": len(dentro),
            "recortes": sum(len(g["recortes"]) for g in dentro),
            "detalle": dentro,
        },
    }
    return quitar, resumen


# --- recortes.csv --------------------------------------------------------------------------


def _numero_texto(valor: float) -> str:
    """Número con hasta dos decimales y sin ceros de sobra (10, 20.5, 33.25)."""
    texto = f"{float(valor):.2f}".rstrip("0").rstrip(".")
    return "0" if texto in ("", "-0") else texto


def tabla_recortes(recortes: list[Candidato], tabla: dict) -> pd.DataFrame:
    """Una fila de ``recortes.csv`` por recorte, con todas las columnas como texto.

    ``salida_global`` sale de la regla de agregación existente (``aggregate``) con las seis
    etiquetas del recorte; ``dhash`` es la huella directa en 16 cifras hexadecimales.
    """
    filas = []
    for c in recortes:
        x, y, w, h = c.caja
        fila = {
            "recorte_id": c.recorte_id,
            "fuente": c.fuente,
            "licencia": c.licencia,
            "imagen_origen": c.imagen_origen,
            "caja_x": _numero_texto(x),
            "caja_y": _numero_texto(y),
            "caja_w": _numero_texto(w),
            "caja_h": _numero_texto(h),
            "origen_caja": c.origen_caja,
            "lechuga_id": c.lechuga_id,
            "vista": c.vista,
            "particion": c.particion,
            **c.etiquetas,
            "etiqueta_original": c.etiqueta_original,
            "salida_global": aggregate(c.etiquetas, tabla).level,
            "dhash": f"{c.huellas[0]:016x}",
        }
        filas.append(fila)
    return pd.DataFrame(filas, columns=list(COLUMNAS))


def validar_recortes(df: pd.DataFrame) -> list[str]:
    """Comprueba una tabla de recortes y lista los problemas (vacía = válida).

    En lenguaje llano: las columnas son exactamente las de ``COLUMNAS``; ``recorte_id`` es único y
    tiene la forma ``<fuente>_<nnnnnn>``; fuente, licencia, imagen de origen y lechuga no van
    vacías; la imagen de origen es solo un nombre de archivo (sin carpetas ni rutas); la partición
    es train, valid o test; la vista va vacía o es una de ``VISTAS``; cada grupo tiene una
    etiqueta suya o ``NO_VISIBLE``; ``salida_global`` es uno de ``GLOBAL_LEVELS``; la caja tiene
    números con ancho y alto positivos y ``dhash`` 16 cifras hexadecimales.
    """
    faltan = [c for c in COLUMNAS if c not in df.columns]
    sobran = [c for c in df.columns if c not in COLUMNAS]
    if faltan or sobran:
        return [f"columnas incorrectas: faltan {faltan}, sobran {sobran}"]
    errores = []
    for recorte_id, veces in Counter(df["recorte_id"]).items():
        if veces > 1:
            errores.append(f"recorte_id '{recorte_id}' repetido ({veces} filas)")
    for posicion, fila in enumerate(df.to_dict("records"), start=1):
        quien = f"fila {posicion} ({fila['recorte_id']})"
        if not _RECORTE_ID.fullmatch(str(fila["recorte_id"])):
            errores.append(f"{quien}: recorte_id no tiene la forma <fuente>_<nnnnnn>")
        for columna in ("fuente", "licencia", "imagen_origen", "lechuga_id"):
            if is_blank(fila[columna]):
                errores.append(f"{quien}: '{columna}' vacía")
        origen = str(fila["imagen_origen"])
        if "/" in origen or "\\" in origen or contiene_ruta_absoluta(origen):
            errores.append(f"{quien}: imagen_origen '{origen}' debe ser un nombre de archivo, "
                           "sin carpetas")  # fmt: skip
        if fila["particion"] not in PARTICIONES:
            errores.append(f"{quien}: particion '{fila['particion']}' no válida "
                           f"(válidas: {', '.join(PARTICIONES)})")  # fmt: skip
        if not is_blank(fila["vista"]) and fila["vista"] not in VISTAS:
            errores.append(f"{quien}: vista '{fila['vista']}' no válida "
                           f"(válidas: {', '.join(VISTAS)} o vacía)")  # fmt: skip
        for grupo, validas in GROUPS.items():
            if fila[grupo] not in (*validas, NO_VISIBLE):
                errores.append(f"{quien}: valor '{fila[grupo]}' no válido en '{grupo}'")
        if fila["salida_global"] not in GLOBAL_LEVELS:
            errores.append(f"{quien}: salida_global '{fila['salida_global']}' no válida")
        if fila["origen_caja"] not in ORIGENES:
            errores.append(f"{quien}: origen_caja '{fila['origen_caja']}' no válido")
        try:
            caja = [float(fila[c]) for c in ("caja_x", "caja_y", "caja_w", "caja_h")]
            if caja[2] <= 0 or caja[3] <= 0:
                errores.append(f"{quien}: la caja tiene ancho o alto no positivo")
        except (TypeError, ValueError):
            errores.append(f"{quien}: la caja no son números")
        if not _DHASH.fullmatch(str(fila["dhash"])):
            errores.append(f"{quien}: dhash debe tener 16 cifras hexadecimales")
    return errores


# --- Informe -------------------------------------------------------------------------------


def _distribucion(recortes: list[Candidato], orden_fuentes: list[str]) -> dict:
    """Recortes por fuente y partición, con cuántos hay de cada etiqueta de cada grupo."""
    resultado = {}
    for nombre in orden_fuentes:
        por_particion = {}
        for particion in PARTICIONES:
            de_aqui = [c for c in recortes if c.fuente == nombre and c.particion == particion]
            etiquetas = {g: dict.fromkeys((*GROUPS[g], NO_VISIBLE), 0) for g in GROUPS}
            for c in de_aqui:
                for grupo, etiqueta in c.etiquetas.items():
                    etiquetas[grupo][etiqueta] += 1
            por_particion[particion] = {"recortes": len(de_aqui), "etiquetas": etiquetas}
        resultado[nombre] = por_particion
    return resultado


def construir_informe(
    config: ConfigEtapa2,
    seleccion: Seleccion,
    recortes: list[Candidato],
    *,
    comprobar: bool,
    propio: dict | None = None,
    duplicados: dict | None = None,
    cajas_raras: list[dict] = (),
    avisos: list[str] = (),
) -> dict:
    """Resumen del dataset, listo para guardar como JSON (sin fechas: así es idempotente).

    Con ``comprobar`` cuenta los candidatos (antes de recortar y de quitar duplicados): no hay
    detecciones ni duplicados, que necesitan los píxeles.
    """
    orden = [f.nombre for f in config.fuentes]
    if propio is None:
        propio = {f.nombre: {"fotos": sum(1 for c in recortes if c.fuente == f.nombre),
                             "sin_deteccion": None, "detecciones": None}
                  for f in config.fuentes if isinstance(f, FuentePropia)}  # fmt: skip
    return {
        "config": config.config_rel,
        "salida": config.salida_rel,
        "semilla": config.semilla,
        "comprobar": comprobar,
        "parametros": {
            "recorte": {
                "margen": config.recorte.margen,
                "lado_px": config.recorte.lado_px,
                "calidad_jpeg": config.recorte.calidad_jpeg,
                "min_lado_px": config.recorte.min_lado_px,
            },
            "duplicados": {
                "umbral_hamming": config.duplicados.umbral_hamming,
                "giros_y_espejos": config.duplicados.giros_y_espejos,
                "prioridad_particiones": list(config.duplicados.prioridad_particiones),
            },
        },
        "recortes": _distribucion(recortes, orden),
        "totales": {p: sum(1 for c in recortes if c.particion == p) for p in PARTICIONES},
        "propio": propio,
        "tope_bio_ok": seleccion.tope,
        "plantulas_excluidas": seleccion.plantulas,
        "cajas_raras": [*seleccion.cajas_raras, *cajas_raras],
        "reparto_por_grupo": seleccion.reparto,
        "duplicados": duplicados,
        "avisos": [*seleccion.avisos, *avisos],
    }


def formatear_informe(informe: dict) -> str:
    """Texto legible del informe para la consola (el detalle completo está en el JSON)."""
    titulo = f"Dataset de recortes de la etapa 2 (semilla {informe['semilla']})"
    if informe["comprobar"]:
        titulo += " -- comprobar: candidatos, antes de recortar y de quitar duplicados"
    lineas = [titulo, "  recortes por fuente y partición:"]
    for nombre, por_particion in informe["recortes"].items():
        lineas.append(f"    {nombre}:")
        for particion, datos in por_particion.items():
            bio = {e: n for e, n in datos["etiquetas"]["daño_bio"].items() if n}
            detalle = f" (daño_bio: {', '.join(f'{e} {n}' for e, n in bio.items())})" if bio else ""
            lineas.append(f"      {particion}: {datos['recortes']}{detalle}")
    lineas.append("  total: " + ", ".join(f"{p} {n}" for p, n in informe["totales"].items()))
    for nombre, datos in informe["propio"].items():
        if datos["sin_deteccion"] is not None:
            lineas.append(f"  fotos propias sin detección ({nombre}): "
                          f"{len(datos['sin_deteccion'])} de {datos['fotos']}")  # fmt: skip
    for nombre, por_particion in informe["tope_bio_ok"].items():
        partes = [f"{p} {d['candidatos']} -> {d['elegidos']}" for p, d in por_particion.items()
                  if d["tope"] is not None]  # fmt: skip
        if partes:
            lineas.append(f"  tope de recortes sin defecto ({nombre}): {', '.join(partes)}")
    for nombre, datos in informe["plantulas_excluidas"].items():
        lineas.append(f"  imágenes excluidas por plántula ({nombre}): {datos['imagenes']}")
    raras = informe["cajas_raras"]
    descartadas = sum(r["descartada"] for r in raras)
    lineas.append(f"  cajas raras: {len(raras)} ({descartadas} descartadas)")
    motivos = Counter(r["motivo"].split(":")[0] for r in raras)
    lineas += [f"    {motivo}: {n}" for motivo, n in sorted(motivos.items())]
    for nombre, por_particion in informe["reparto_por_grupo"].items():
        partes = [f"{p} {len(d['grupos'])} ids ({len(d['grupos_con_defecto'])} con defecto, "
                  f"{d['imagenes']} imágenes)" for p, d in por_particion.items()]  # fmt: skip
        lineas.append(f"  reparto por id ({nombre}): {'; '.join(partes)}")
    duplicados = informe["duplicados"]
    if duplicados is not None:
        giros = "con" if duplicados["giros_y_espejos"] else "sin"
        lineas.append(
            f"  duplicados (Hamming <= {duplicados['umbral_hamming']}, {giros} giros y espejos): "
            f"{duplicados['grupos_que_cruzan']} grupos cruzan particiones o fuentes, "
            f"{duplicados['recortes_quitados']} recortes quitados; "
            f"{duplicados['dentro_de_una_celda']['grupos']} grupos dentro de una misma partición "
            "y fuente (no se quitan)"
        )
    lineas += [f"  AVISO: {aviso}" for aviso in informe["avisos"]]
    return "\n".join(lineas)


# --- Orquestación --------------------------------------------------------------------------


def _comprobar_salida(carpeta: Path, nombre: str) -> None:
    """Error si la carpeta existe y tiene algo que no haya escrito este script (no se borra)."""
    if not carpeta.exists():
        return
    if not carpeta.is_dir():
        raise ValueError(f"La salida {nombre} existe y no es una carpeta")
    ajenos = sorted(p.name for p in carpeta.iterdir() if p.name not in _CONTENIDO_PROPIO)
    if ajenos:
        raise ValueError(
            f"No se toca la salida {nombre}: contiene archivos que no son de este script "
            f"({', '.join(ajenos)}). Muévelos o elige otra carpeta de salida."
        )


def construir_dataset(
    ruta_config: str | Path,
    *,
    raiz: str | Path,
    tabla: dict,
    cargador_detector: Callable[[Path, str], object] | None = None,
    comprobar: bool = False,
    progreso: Callable[[str], None] | None = None,
) -> dict:
    """Construye el dataset de recortes de la etapa 2 y devuelve el informe.

    En lenguaje llano, en este orden: valida el YAML; comprueba que la salida (y una posible
    ``<salida>.parcial`` de un fallo anterior) solo tiene lo que escribe este script y que existen
    los pesos de RF-DETR; junta los candidatos. Con ``comprobar`` para aquí y devuelve los
    recuentos sin cargar el detector ni escribir nada. Si no, recorta todo en
    ``<salida>.parcial``, quita los duplicados, valida la tabla y escribe ``recortes.csv`` e
    ``informe.json``; solo entonces borra la salida anterior y pone la nueva en su lugar. Si algo
    falla por el camino, se borra la ``.parcial`` y la salida anterior queda intacta.

    ``tabla`` es la de ``configs/salida_global.yaml``. ``cargador_detector(ruta, dispositivo)``
    sustituye la carga real de RF-DETR al probar. ``progreso`` recibe textos de avance.
    """
    raiz = Path(raiz)
    config = cargar_config(ruta_config, raiz)
    parcial = config.salida.with_name(config.salida.name + ".parcial")
    _comprobar_salida(config.salida, config.salida_rel)
    _comprobar_salida(parcial, f"{config.salida_rel}.parcial")
    propias = [f for f in config.fuentes if isinstance(f, FuentePropia)]
    if cargador_detector is None:
        for fuente in propias:
            if not (raiz / fuente.detector.pesos).is_file():
                raise FileNotFoundError(f"Fuente '{fuente.nombre}': no existen los pesos de "
                                        f"RF-DETR: {fuente.detector.pesos}")  # fmt: skip

    seleccion = seleccionar(config, progreso)
    if comprobar:
        return construir_informe(config, seleccion, seleccion.candidatos, comprobar=True)

    cargar = cargador_detector or cargar_rfdetr
    predictores = {}
    for fuente in propias:
        modelo = cargar(raiz / fuente.detector.pesos, fuente.detector.dispositivo)
        predictores[fuente.nombre] = predictor_rfdetr(modelo, fuente.detector.umbral)

    if parcial.exists():
        shutil.rmtree(parcial)
    try:
        (parcial / "imagenes").mkdir(parents=True)
        (parcial / "control").mkdir()
        hechos, propio, raras, avisos = materializar(seleccion.candidatos, config, parcial,
                                                     predictores, progreso)  # fmt: skip
        if progreso:
            progreso(f"buscando duplicados entre {len(hechos)} recortes")
        orden = [f.nombre for f in config.fuentes]
        quitar, duplicados = resolver_duplicados(hechos, config.duplicados, orden)
        for recorte_id in sorted(quitar):
            (parcial / "imagenes" / f"{recorte_id}.jpg").unlink()
        finales = sorted((c for c in hechos if c.recorte_id not in quitar),
                         key=lambda c: (orden.index(c.fuente), c.recorte_id))  # fmt: skip
        df = tabla_recortes(finales, tabla)
        errores = validar_recortes(df)
        if errores:
            raise ValueError("recortes.csv no es válido:\n  " + "\n  ".join(errores[:10]))
        informe = construir_informe(
            config,
            seleccion,
            finales,
            comprobar=False,
            propio=propio,
            duplicados=duplicados,
            cajas_raras=raras,
            avisos=avisos,
        )
        df.to_csv(parcial / "recortes.csv", index=False, encoding="utf-8", lineterminator="\n")
        (parcial / "informe.json").write_text(
            json.dumps(informe, ensure_ascii=False, indent=1), encoding="utf-8"
        )
    except BaseException:
        shutil.rmtree(parcial, ignore_errors=True)
        raise
    if config.salida.exists():
        shutil.rmtree(config.salida)
    parcial.rename(config.salida)
    return informe
