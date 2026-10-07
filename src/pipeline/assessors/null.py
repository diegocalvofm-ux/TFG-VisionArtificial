"""Asesor nulo de la v0: todavía no hay modelo de calidad."""

import numpy as np

from src.data.schema import GROUPS, NO_VISIBLE
from src.pipeline.types import Detection


class NullAssessor:
    """No evalúa nada: declara los seis grupos como NO_VISIBLE."""

    def assess(self, imagen: np.ndarray, deteccion: Detection) -> dict[str, str]:
        """Devuelve ``{grupo: NO_VISIBLE}`` para los seis grupos del esquema, en su orden."""
        return {grupo: NO_VISIBLE for grupo in GROUPS}
