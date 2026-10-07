"""Detector trivial de la v0: toda la imagen es una sola lechuga."""

import numpy as np

from src.pipeline.types import Detection


class WholeDetector:
    """Considera que la imagen completa es la lechuga.

    Sirve para probar el resto del pipeline sin ningún modelo: siempre devuelve una
    detección, con puntuación 1.0.
    """

    def detect(self, imagen: np.ndarray) -> list[Detection]:
        """Devuelve una única caja que cubre la imagen entera: (0, 0, ancho, alto)."""
        alto, ancho = imagen.shape[:2]
        return [Detection(box=(0, 0, ancho, alto), score=1.0)]
