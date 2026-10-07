"""Tipos comunes del pipeline: detecciones y contratos de detector y asesor.

Las imágenes son arrays de numpy en formato BGR de OpenCV, con forma (alto, ancho, 3).
"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class Detection:
    """Una lechuga localizada en una imagen.

    ``box`` son las esquinas de la caja en píxeles: (x1, y1, x2, y2), con (0, 0) arriba a
    la izquierda. ``score`` es la confianza del detector (de 0 a 1).
    """

    box: tuple[int, int, int, int]
    score: float
    label: str = "lettuce"


class Detector(Protocol):
    """Etapa 1: localiza las lechugas de una imagen."""

    def detect(self, imagen: np.ndarray) -> list[Detection]: ...


class Assessor(Protocol):
    """Etapa 2: da una etiqueta a cada uno de los seis grupos de una lechuga detectada."""

    def assess(self, imagen: np.ndarray, deteccion: Detection) -> dict[str, str]: ...
