"""Registro de detectores: asocia el nombre que se pide por línea de comandos con su clase."""

from collections.abc import Callable, Mapping

from src.pipeline.detectors.whole import WholeDetector
from src.pipeline.types import Detector


def _crear_whole(cfg: Mapping) -> Detector:
    return WholeDetector()


# Nombre -> fábrica. Cada fábrica recibe la configuración (configs/v0.yaml) para poder
# leer sus parámetros; así se añaden detectores nuevos sin cambiar esta interfaz.
DETECTORS: dict[str, Callable[[Mapping], Detector]] = {"whole": _crear_whole}


def nombres_detectores() -> list[str]:
    """Nombres de los detectores disponibles, ordenados."""
    return sorted(DETECTORS)


def crear_detector(nombre: str, cfg: Mapping) -> Detector:
    """Crea el detector con ese nombre.

    Error: ``ValueError`` que lista los disponibles si el nombre no está registrado.
    """
    if nombre not in DETECTORS:
        raise ValueError(
            f"Detector '{nombre}' desconocido. Disponibles: {', '.join(nombres_detectores())}"
        )
    return DETECTORS[nombre](cfg)
