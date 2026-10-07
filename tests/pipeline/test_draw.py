"""Pruebas del dibujo de la imagen anotada."""

import numpy as np

from src.data.schema import GLOBAL_LEVELS
from src.pipeline.draw import COLORES, dibujar, texto_ascii


def _lechuga(caja, nivel="REVIEW"):
    return {"box": list(caja), "score": 1.0, "global": nivel, "provisional_labels": []}


def test_hay_un_color_bgr_distinto_por_cada_nivel():
    assert len(COLORES) == len(GLOBAL_LEVELS)
    assert len(set(COLORES)) == len(COLORES)
    for color in COLORES:
        assert len(color) == 3 and all(0 <= canal <= 255 for canal in color)


def test_texto_ascii_quita_lo_que_opencv_no_pinta():
    assert texto_ascii("daño ñandú").isascii()
    assert texto_ascii("REVIEW 1.00") == "REVIEW 1.00"


def test_dibujar_devuelve_una_copia_distinta_y_no_toca_la_original(crear_imagen):
    original = crear_imagen(100, 120)
    antes = original.copy()

    dibujada = dibujar(original, [_lechuga((10, 10, 100, 80))])

    np.testing.assert_array_equal(original, antes)  # la entrada no cambia
    assert dibujada is not original
    assert dibujada.shape == original.shape and dibujada.dtype == original.dtype
    assert not np.array_equal(dibujada, original)


def test_dibujar_pinta_el_borde_de_la_caja_con_el_color_del_nivel(crear_imagen):
    original = crear_imagen(100, 120)

    for nivel in GLOBAL_LEVELS:
        dibujada = dibujar(original, [_lechuga((10, 10, 100, 80), nivel)])
        color = COLORES[GLOBAL_LEVELS.index(nivel)]
        # Píxel en mitad del lado izquierdo de la caja: lejos del texto.
        assert tuple(dibujada[45, 10]) == color, nivel


def test_dibujar_sin_lechugas_devuelve_una_copia_igual(crear_imagen):
    original = crear_imagen(100, 120)

    dibujada = dibujar(original, [])

    assert dibujada is not original
    np.testing.assert_array_equal(dibujada, original)


def test_dibujar_admite_cajas_que_se_salen_de_la_imagen(crear_imagen):
    original = crear_imagen(50, 60)

    dibujada = dibujar(original, [_lechuga((-10, -10, 500, 500))])

    assert dibujada.shape == original.shape
