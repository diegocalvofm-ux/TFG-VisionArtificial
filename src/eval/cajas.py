"""IoU entre cajas y emparejamiento voraz por confianza entre predicciones y referencias.

Las cajas son ``(x1, y1, x2, y2)`` en píxeles, con (0, 0) arriba a la izquierda.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


def _areas(cajas: np.ndarray) -> np.ndarray:
    """Área de cada caja (0 si no tiene ancho o alto)."""
    return np.clip(cajas[:, 2] - cajas[:, 0], 0, None) * np.clip(cajas[:, 3] - cajas[:, 1], 0, None)


def matriz_iou(cajas_a, cajas_b) -> np.ndarray:
    """IoU de cada caja de ``cajas_a`` con cada una de ``cajas_b``: matriz (len(a), len(b)).

    En lenguaje llano: el IoU ("intersección sobre unión") mide cuánto se parecen dos cajas: el
    área que tienen en común dividida por el área que cubren entre las dos. Vale 1 si coinciden
    y 0 si no se tocan (o si se tocan solo por el borde, o una de ellas no tiene área).
    """
    a = np.asarray(cajas_a, dtype=float).reshape(-1, 4)
    b = np.asarray(cajas_b, dtype=float).reshape(-1, 4)
    ancho = np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0])
    alto = np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1])
    comun = np.clip(ancho, 0, None) * np.clip(alto, 0, None)
    union = _areas(a)[:, None] + _areas(b)[None, :] - comun
    return np.divide(comun, union, out=np.zeros_like(union), where=union > 0)


def iou(caja_a: Sequence[float], caja_b: Sequence[float]) -> float:
    """IoU de dos cajas (ver ``matriz_iou``)."""
    return float(matriz_iou([caja_a], [caja_b])[0, 0])


@dataclass(frozen=True)
class Emparejamiento:
    """Resultado de emparejar las predicciones de una imagen con sus cajas de referencia.

    ``pares`` son tuplas ``(índice de la predicción, índice de la referencia, IoU)`` de los
    verdaderos positivos; ``fp`` son los índices de las predicciones sin pareja (falsos
    positivos) y ``fn`` los de las referencias sin pareja (falsos negativos).
    """

    pares: list[tuple[int, int, float]]
    fp: list[int]
    fn: list[int]


def emparejar(cajas_pred, confianzas, cajas_ref, iou_min: float) -> Emparejamiento:
    """Empareja de forma voraz las predicciones con las referencias, de más a menos confianza.

    En lenguaje llano: se recorren las predicciones de la más segura a la menos segura. Cada una
    se queda con la caja de referencia LIBRE con la que más se solapa, siempre que el IoU sea al
    menos ``iou_min``; si no hay ninguna, es un falso positivo. Una referencia solo se empareja una
    vez, así que si dos predicciones cubren el mismo objeto, la segunda es un falso positivo. Las
    referencias que quedan libres son falsos negativos. En un empate de confianza gana la
    predicción que viene antes en la lista.

    Error: ``ValueError`` si hay distinto número de cajas y de confianzas.
    """
    pred = np.asarray(cajas_pred, dtype=float).reshape(-1, 4)
    confianzas = np.asarray(confianzas, dtype=float).reshape(-1)
    ref = np.asarray(cajas_ref, dtype=float).reshape(-1, 4)
    if len(confianzas) != len(pred):
        raise ValueError(
            f"Debe haber una confianza por predicción ({len(confianzas)} confianzas "
            f"para {len(pred)} cajas)"
        )

    matriz = matriz_iou(pred, ref)
    libres = np.ones(len(ref), dtype=bool)
    pares: list[tuple[int, int, float]] = []
    fp: list[int] = []
    for i in np.argsort(-confianzas, kind="stable"):
        candidatas = np.where(libres, matriz[i], -1.0) if len(ref) else np.empty(0)
        mejor = int(np.argmax(candidatas)) if len(ref) else -1
        if mejor >= 0 and libres[mejor] and matriz[i, mejor] >= iou_min:
            libres[mejor] = False
            pares.append((int(i), mejor, float(matriz[i, mejor])))
        else:
            fp.append(int(i))
    return Emparejamiento(pares=pares, fp=sorted(fp), fn=[int(j) for j in np.flatnonzero(libres)])
