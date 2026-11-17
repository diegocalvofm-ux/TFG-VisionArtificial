"""Datos del entrenamiento de la etapa 2: ``recortes.csv``, reparto, smoke y sobremuestreo.

El dataset de la v2-01 (``data/processed/etapa2_v1``) solo se lee: aquí nada se copia ni se
modifica. Todo lo que se hace (repartir de nuevo la fuente propia, quedarse con un subconjunto en
el smoke, repetir los recortes propios en train) se decide con tablas e índices en memoria.

Sin torch: estas funciones se prueban sin instalarlo.
"""

import random
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.dataset_etapa2 import validar_recortes
from src.data.publicos import PARTICION_SALIDA
from src.data.schema import GROUPS, NO_VISIBLE
from src.data.split import _repartir, group_split

PARTICIONES = ("train", "valid", "test")
NOMBRE_CSV = "recortes.csv"
CARPETA_IMAGENES = "imagenes"
_MAX_EN_MENSAJE = 10  # cuántos nombres se citan como mucho en un mensaje de error


def leer_recortes(carpeta: str | Path) -> tuple[pd.DataFrame, list[str]]:
    """Lee ``recortes.csv`` de ``carpeta``, lo valida y añade la ruta de cada imagen.

    En lenguaje llano: carga la tabla (todo como texto, en utf-8) y la valida con las mismas
    reglas que la escribieron (``validar_recortes``: columnas, ids, etiquetas de ``schema.py``...).
    Comprueba que cada recorte tiene su imagen ``imagenes/<recorte_id>.jpg``. Las filas con los
    seis grupos en NO_VISIBLE no enseñan nada al modelo ni se pueden evaluar: se quitan.

    Devuelve ``(tabla, quitadas)``: la tabla con una columna más, ``ruta`` (la imagen), y un
    índice limpio (0, 1, 2...), y la lista de ``recorte_id`` quitados por no tener ningún grupo
    visible. Errores: ``FileNotFoundError`` si falta el CSV o alguna imagen (dice cuántas y
    cuáles); ``ValueError`` si la tabla no es válida.
    """
    carpeta = Path(carpeta)
    ruta_csv = carpeta / NOMBRE_CSV
    if not ruta_csv.is_file():
        raise FileNotFoundError(
            f"No existe {carpeta.name}/{NOMBRE_CSV}: genera el dataset con `python -m "
            "src.data.etapa2 --config configs/datasets/etapa2_v1.yaml` (en Colab, descomprime "
            "etapa2_v1.zip)"
        )
    tabla = pd.read_csv(ruta_csv, dtype=str, keep_default_na=False, encoding="utf-8")
    errores = validar_recortes(tabla)
    if errores:
        detalle = "\n  ".join(errores[:_MAX_EN_MENSAJE])
        raise ValueError(f"{NOMBRE_CSV} no es válido ({len(errores)} problemas):\n  {detalle}")

    tabla["ruta"] = [carpeta / CARPETA_IMAGENES / f"{r}.jpg" for r in tabla["recorte_id"]]
    faltan = [ruta.name for ruta in tabla["ruta"] if not ruta.is_file()]
    if faltan:
        mas = " ..." if len(faltan) > _MAX_EN_MENSAJE else ""
        raise FileNotFoundError(
            f"Faltan {len(faltan)} imágenes en {carpeta.name}/{CARPETA_IMAGENES}: "
            f"{', '.join(faltan[:_MAX_EN_MENSAJE])}{mas}"
        )

    sin_etiquetas = (tabla[list(GROUPS)] == NO_VISIBLE).all(axis=1)
    quitadas = tabla.loc[sin_etiquetas, "recorte_id"].tolist()
    return tabla[~sin_etiquetas].reset_index(drop=True), quitadas


def codificar(tabla: pd.DataFrame) -> np.ndarray:
    """Las seis etiquetas de cada fila como números: matriz (filas, 6) de enteros.

    En lenguaje llano: la columna ``j`` es el ``j``-ésimo grupo de ``GROUPS`` y el valor es la
    posición de la etiqueta en ``GROUPS[grupo]`` (``CUT_OK`` = 0, ``STEM_TOO_LONG`` = 1...).
    NO_VISIBLE es −1: la pérdida y las métricas ignoran esas casillas.
    """
    salida = np.full((len(tabla), len(GROUPS)), -1, dtype=np.int64)
    for j, (grupo, etiquetas) in enumerate(GROUPS.items()):
        posicion = {etiqueta: i for i, etiqueta in enumerate(etiquetas)}
        for i, valor in enumerate(tabla[grupo]):
            if valor != NO_VISIBLE:
                salida[i, j] = posicion[valor]
    return salida


def repartir_propio(
    tabla: pd.DataFrame, fuente: str, fracciones: tuple[float, float, float], semilla: int
) -> tuple[pd.DataFrame, dict]:
    """Reparte de nuevo las filas de ``fuente`` en train, valid y test por ``lechuga_id``.

    En lenguaje llano: es para cuando haya más lechugas propias. Todas las fotos (vistas y
    estados) de una misma lechuga van a la misma partición, con ``group_split`` (el mismo reparto
    por lechuga del resto del proyecto, con semilla); las fracciones se cumplen en número de
    lechugas. Las filas de las demás fuentes conservan la partición de ``recortes.csv``. La tabla
    de entrada no se modifica.

    Devuelve ``(tabla nueva, resumen)`` con las fracciones y las lechugas de cada partición.
    Errores (``ValueError``): no hay filas de ``fuente`` o hay menos de 3 lechugas.
    """
    propias = tabla["fuente"] == fuente
    if not propias.any():
        raise ValueError(f"datos.reparto_propio: no hay recortes de la fuente '{fuente}'")
    try:
        repartidas = group_split(tabla[propias], "lechuga_id", fracciones, semilla)
    except ValueError as error:
        raise ValueError(
            f"datos.reparto_propio: no se pueden repartir las lechugas de '{fuente}': {error}"
        ) from error
    salida = tabla.copy()
    salida.loc[propias, "particion"] = [PARTICION_SALIDA[p] for p in repartidas["split"]]
    lechugas = {
        particion: sorted(set(salida.loc[propias & (salida["particion"] == particion),
                                         "lechuga_id"]))  # fmt: skip
        for particion in PARTICIONES
    }
    resumen = {"fracciones": dict(zip(PARTICIONES, fracciones, strict=True)), "lechugas": lechugas}
    return salida, resumen


def subconjunto(tabla: pd.DataFrame, n: int, semilla: int) -> pd.DataFrame:
    """``n`` filas de ``tabla`` para el modo smoke, repartidas entre las fuentes.

    En lenguaje llano: a cada fuente le tocan filas en proporción a las que tiene, pero toda fuente
    presente recibe al menos una (si ``n`` alcanza): así el smoke también pasa por la fuente
    propia, la única con etiquetas en los cinco primeros grupos. Dentro de cada fuente las filas
    se sortean con la semilla, en un orden fijo (siempre las mismas). Si ``n`` no es menor que el
    total, se devuelve la tabla entera.
    """
    if n >= len(tabla):
        return tabla.reset_index(drop=True)
    fuentes = list(dict.fromkeys(tabla["fuente"]))  # en el orden en que aparecen
    tamanos = [int((tabla["fuente"] == f).sum()) for f in fuentes]
    if n >= len(fuentes):
        cuotas = _repartir(n, [t / len(tabla) for t in tamanos])
    else:  # no da para una por fuente: proporcional sin mínimo
        cuotas = [round(n * t / len(tabla)) for t in tamanos]
    generador = random.Random(semilla)
    elegidos = []
    for fuente, cuota in zip(fuentes, cuotas, strict=True):
        ids = sorted(tabla.loc[tabla["fuente"] == fuente, "recorte_id"])
        elegidos += generador.sample(ids, min(cuota, len(ids)))
    elegidos = set(elegidos)
    return tabla[tabla["recorte_id"].isin(elegidos)].reset_index(drop=True)


def indices_por_epoca(train: pd.DataFrame, fuente: str, factor: int) -> list[int]:
    """Las filas de train de una época, con las de ``fuente`` repetidas ``factor`` veces.

    En lenguaje llano: es el sobremuestreo. Cada fila de train aparece una vez y cada fila de la
    fuente propia, ``factor`` veces (1 = sin repetir); cada repetición recibe aumentos distintos.
    Solo admite filas de train: valid y test nunca se repiten. Devuelve las posiciones (0, 1...)
    en ``train``, ordenadas.
    """
    if isinstance(factor, bool) or not isinstance(factor, int) or factor < 1:
        raise ValueError(f"sobremuestreo: debe ser un entero >= 1 (llegó {factor!r})")
    if (train["particion"] != "train").any():
        raise ValueError("sobremuestreo: solo train (valid y test nunca se repiten)")
    indices = []
    for posicion, nombre in enumerate(train["fuente"]):
        indices += [posicion] * (factor if nombre == fuente else 1)
    return indices


def recuentos(tabla: pd.DataFrame) -> dict:
    """Recortes por partición y fuente, con cuántos hay de cada etiqueta de cada grupo.

    Forma: ``{partición: {fuente: {"recortes": n, "etiquetas": {grupo: {etiqueta: n}}}}}``. Una
    fuente sin filas en una partición no aparece en ella. Sirve para ``datos.json``.
    """
    salida = {}
    for particion in PARTICIONES:
        de_aqui = tabla[tabla["particion"] == particion]
        por_fuente = {}
        for fuente in dict.fromkeys(de_aqui["fuente"]):
            filas = de_aqui[de_aqui["fuente"] == fuente]
            etiquetas = {}
            for grupo, validas in GROUPS.items():
                cuenta = Counter(filas[grupo])
                etiquetas[grupo] = {e: cuenta.get(e, 0) for e in (*validas, NO_VISIBLE)}
            por_fuente[fuente] = {"recortes": len(filas), "etiquetas": etiquetas}
        salida[particion] = por_fuente
    return salida
