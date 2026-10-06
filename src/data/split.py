"""Partición de los datos en train / val / test por lechuga (nunca por imagen)."""

import math
import warnings

import numpy as np
import pandas as pd

from src.data.schema import is_blank

PARTICIONES = ("train", "val", "test")


def _validar_fracciones(fracciones) -> None:
    """Lanza ValueError si las fracciones no son tres, positivas y con suma 1."""
    if len(fracciones) != len(PARTICIONES):
        raise ValueError(
            f"Se esperan {len(PARTICIONES)} fracciones (train, val, test) y llegaron "
            f"{len(fracciones)}: {tuple(fracciones)}"
        )
    if any(f <= 0 for f in fracciones):
        raise ValueError(f"Las fracciones deben ser positivas: {tuple(fracciones)}")
    if not math.isclose(sum(fracciones), 1.0, abs_tol=1e-9):
        raise ValueError(f"Las fracciones deben sumar 1 y suman {sum(fracciones)}: {fracciones}")


def _repartir(n_grupos: int, fracciones) -> list[int]:
    """Reparte n_grupos entre las particiones según las fracciones, sin dejar ninguna vacía.

    En lenguaje llano: primero se da a cada partición la parte entera de lo que le toca
    (p. ej. 70.0 de 100). Los grupos que sobran por el redondeo se dan, uno a uno, a las
    particiones a las que les quedó más decimal ("método del mayor resto"). Por último, si
    alguna partición se quedó en 0, se le quita un grupo a la más grande.
    """
    exactos = [f * n_grupos for f in fracciones]
    cantidades = [math.floor(x) for x in exactos]
    restos = [x - c for x, c in zip(exactos, cantidades, strict=True)]
    sobran = n_grupos - sum(cantidades)
    for i in sorted(range(len(cantidades)), key=lambda i: (-restos[i], i))[:sobran]:
        cantidades[i] += 1
    for i, cantidad in enumerate(cantidades):
        if cantidad == 0:
            donante = max(range(len(cantidades)), key=lambda j: cantidades[j])
            cantidades[donante] -= 1
            cantidades[i] += 1
    return cantidades


def group_split(
    df: pd.DataFrame,
    group_col: str = "lechuga_id",
    fracciones: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
) -> pd.DataFrame:
    """Asigna cada fila a train, val o test haciendo que cada lechuga caiga en una sola.

    En lenguaje llano: todas las fotos de una misma lechuga acaban en la misma partición.
    Así el modelo nunca se evalúa con fotos de una planta que ya vio al entrenar (fuga de
    datos). Se baraja la lista de lechugas con la semilla dada y se reparte según las
    fracciones; el resultado es siempre el mismo con la misma semilla.

    Las fracciones se cumplen en número de lechugas, no de imágenes (cada lechuga puede
    tener un número distinto de fotos). No se estratifica por etiqueta.

    Las filas sin ``group_col`` (vacías) forman cada una su propio grupo y se marcan en
    ``grupo_inferido``; se emite un aviso porque, en datasets públicos, eso no impide que
    fotos de la misma planta caigan en particiones distintas.

    Devuelve una copia de ``df`` con dos columnas nuevas: ``split`` ("train", "val" o
    "test") y ``grupo_inferido`` (True si la fila no tenía id). ``df`` no se modifica.
    Los ids se comparan como texto, así que ``1`` y ``"1"`` son la misma lechuga.

    Errores (ValueError): fracciones que no son tres números positivos con suma 1, columna
    de grupo inexistente, o menos de tres grupos (uno por partición).
    """
    _validar_fracciones(fracciones)
    if group_col not in df.columns:
        raise ValueError(f"La columna de grupo '{group_col}' no existe en el DataFrame")

    sin_id = [is_blank(valor) for valor in df[group_col].tolist()]
    # Clave de cada fila: (0, id real) o (1, posición) si no tiene id. El primer elemento
    # evita que un id real pueda confundirse con la clave de una fila sin id.
    claves = [
        (1, f"{pos:09d}") if falta else (0, str(valor))
        for pos, (valor, falta) in enumerate(zip(df[group_col].tolist(), sin_id, strict=True))
    ]
    grupos = sorted(set(claves))  # ordenados, para que la semilla dé siempre lo mismo
    if len(grupos) < len(PARTICIONES):
        raise ValueError(
            f"Hay {len(grupos)} grupos y hacen falta al menos {len(PARTICIONES)} "
            "(uno por partición)"
        )

    n_sin_id = sum(sin_id)
    if n_sin_id:
        warnings.warn(
            f"{n_sin_id} {'fila' if n_sin_id == 1 else 'filas'} sin {group_col}: cada una se "
            "trata como su propia lechuga. En datasets públicos esto no impide fugas entre "
            "fotos de la misma planta: podrían quedar en particiones distintas.",
            UserWarning,
            stacklevel=2,
        )

    orden = np.random.default_rng(seed).permutation(len(grupos))
    asignacion: dict[tuple, str] = {}
    inicio = 0
    for nombre, cantidad in zip(PARTICIONES, _repartir(len(grupos), fracciones), strict=True):
        for posicion in orden[inicio : inicio + cantidad]:
            asignacion[grupos[posicion]] = nombre
        inicio += cantidad

    salida = df.copy()
    salida["split"] = [asignacion[clave] for clave in claves]
    salida["grupo_inferido"] = sin_id
    return salida
