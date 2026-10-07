"""Fixtures compartidas por las pruebas de src/pipeline (imágenes sintéticas, sin datos reales)."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.config import load_config

RAIZ = Path(__file__).resolve().parents[2]


@pytest.fixture
def crear_imagen():
    """Devuelve una función que fabrica una imagen BGR sintética (degradados, sin azar)."""

    def fabrica(alto: int = 48, ancho: int = 64) -> np.ndarray:
        horizontal = np.tile(np.linspace(0, 255, ancho, dtype=np.uint8), (alto, 1))
        vertical = np.tile(np.linspace(255, 0, alto, dtype=np.uint8)[:, None], (1, ancho))
        constante = np.full((alto, ancho), 80, dtype=np.uint8)
        return np.stack([horizontal, vertical, constante], axis=2)

    return fabrica


@pytest.fixture
def escribir_imagen():
    """Devuelve una función que guarda una imagen en disco con imencode + tofile.

    Está escrita aquí a propósito, sin usar src/, para que las pruebas no dependan del
    código que se está probando. La extensión de la ruta decide el formato.
    """

    def escribir(ruta: Path, imagen: np.ndarray) -> Path:
        ruta.parent.mkdir(parents=True, exist_ok=True)
        correcto, datos = cv2.imencode(ruta.suffix, imagen)
        assert correcto
        datos.tofile(ruta)
        return ruta

    return escribir


@pytest.fixture(scope="module")
def tabla():
    """Tabla real de salida global (configs/salida_global.yaml)."""
    return load_config(RAIZ / "configs" / "salida_global.yaml")
