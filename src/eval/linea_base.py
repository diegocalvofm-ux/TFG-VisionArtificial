"""Línea base clásica sin aprendizaje: umbral en HSV, morfología y componentes conexas.

Sirve de referencia para ver cuánto aporta el aprendizaje. Es deliberadamente simple: busca zonas
verdes y toma la caja de cada una. Limitaciones conocidas: lechugas que se tocan se funden en una
sola caja, y una lechuga que no sea verde (roja, o muy oscura o clara) no se ve.
"""

import cv2
import numpy as np

from src.eval.config import ParametrosHSV
from src.pipeline.types import Detection


class LineaBaseHSV:
    """Detector sin aprendizaje con la misma firma que un ``Detector`` del pipeline."""

    def __init__(self, parametros: ParametrosHSV):
        self.parametros = parametros

    def detect(self, imagen: np.ndarray) -> list[Detection]:
        """Devuelve una caja por cada zona verde grande, de mayor a menor área.

        En lenguaje llano, en este orden: 1) se pasa la imagen (BGR) a HSV y se marcan los píxeles
        cuyo tono, saturación y brillo caen en los rangos del YAML (el verde); 2) la APERTURA
        (erosionar y dilatar) borra las motas pequeñas; 3) el CIERRE (dilatar y erosionar) tapa los
        huecos dentro de una lechuga; 4) cada grupo de píxeles conectados es una componente; 5) las
        componentes con menos área que ``area_min_frac`` de la imagen se descartan; 6) la caja de
        cada una es su rectángulo mínimo. No hay confianza: todas llevan puntuación 1,0 de relleno.
        """
        p = self.parametros
        hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
        mascara = cv2.inRange(hsv, (p.h[0], p.s[0], p.v[0]), (p.h[1], p.s[1], p.v[1]))
        mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, _kernel(p.apertura_px))
        mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, _kernel(p.cierre_px))

        n, _, estadisticas, _ = cv2.connectedComponentsWithStats(mascara, connectivity=8)
        alto, ancho = imagen.shape[:2]
        area_minima = p.area_min_frac * alto * ancho
        componentes = [
            estadisticas[i] for i in range(1, n)  # la 0 es el fondo
            if estadisticas[i, cv2.CC_STAT_AREA] >= area_minima
        ]  # fmt: skip
        componentes.sort(key=lambda e: -int(e[cv2.CC_STAT_AREA]))  # estable: empates en orden
        return [
            Detection(
                box=(
                    int(e[cv2.CC_STAT_LEFT]),
                    int(e[cv2.CC_STAT_TOP]),
                    int(e[cv2.CC_STAT_LEFT] + e[cv2.CC_STAT_WIDTH]),
                    int(e[cv2.CC_STAT_TOP] + e[cv2.CC_STAT_HEIGHT]),
                ),
                score=1.0,
            )
            for e in componentes
        ]


def _kernel(lado: int) -> np.ndarray:
    """Kernel elíptico de ``lado`` x ``lado`` píxeles."""
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (lado, lado))
