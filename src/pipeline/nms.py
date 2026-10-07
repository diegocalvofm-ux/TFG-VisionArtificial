"""NMS (non-maximum suppression): quita las cajas repetidas sobre un mismo objeto."""

import numpy as np


def nms(cajas, puntuaciones, iou_umbral: float) -> list[int]:
    """Devuelve los índices de las cajas que se conservan, de mayor a menor puntuación.

    En lenguaje llano: un detector suele marcar varias cajas casi iguales sobre el mismo
    objeto. Se recorre de la mejor puntuación a la peor: cada caja se queda y, a
    continuación, se tiran las que se solapan con ella más que el umbral. Como en el NMS
    estándar, una caja que ya se tiró no vuelve a eliminar a otras.

    El solapamiento es el IoU (área común dividida por el área de la unión). Se suprime
    cuando el IoU es MAYOR que ``iou_umbral``; con un IoU justo igual al umbral se conservan
    las dos. Los empates de puntuación los gana el índice menor.

    ``cajas`` es una matriz (N, 4) con (x1, y1, x2, y2) y ``puntuaciones`` un vector de N.

    Errores (ValueError): formas que no cuadran, o ``iou_umbral`` fuera de [0, 1].
    """
    cajas = np.asarray(cajas, dtype=float)
    puntuaciones = np.asarray(puntuaciones, dtype=float)
    if not 0.0 <= iou_umbral <= 1.0:
        raise ValueError(f"iou_umbral debe estar entre 0 y 1 (recibido: {iou_umbral})")
    if cajas.size == 0 and puntuaciones.size == 0:
        return []
    if cajas.ndim != 2 or cajas.shape[1] != 4:
        raise ValueError(f"cajas debe tener forma (N, 4); recibido: {cajas.shape}")
    if puntuaciones.ndim != 1 or len(puntuaciones) != len(cajas):
        raise ValueError(
            f"puntuaciones debe ser un vector con una puntuación por caja; "
            f"recibido: {puntuaciones.shape} para {len(cajas)} cajas"
        )

    x1, y1, x2, y2 = cajas.T
    areas = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    pendientes = np.argsort(-puntuaciones, kind="stable")  # mejor primero, empates por índice
    conservadas: list[int] = []
    while pendientes.size:
        mejor, resto = pendientes[0], pendientes[1:]
        conservadas.append(int(mejor))
        # Rectángulo común entre la mejor caja y cada una de las restantes.
        ancho = np.minimum(x2[mejor], x2[resto]) - np.maximum(x1[mejor], x1[resto])
        alto = np.minimum(y2[mejor], y2[resto]) - np.maximum(y1[mejor], y1[resto])
        comun = np.clip(ancho, 0, None) * np.clip(alto, 0, None)
        union = areas[mejor] + areas[resto] - comun
        iou = np.divide(comun, union, out=np.zeros_like(union), where=union > 0)
        pendientes = resto[iou <= iou_umbral]
    return conservadas
