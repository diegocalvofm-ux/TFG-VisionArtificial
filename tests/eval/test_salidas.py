"""Pruebas del formato de las salidas: números con coma, CSV y dibujo de cajas."""

import numpy as np

from src.eval.salidas import (
    COLOR_PREDICHA,
    COLOR_REFERENCIA,
    dibujar_cajas,
    escribir_csv,
    formato,
)
from src.pipeline.types import Detection


def test_formato_usa_coma_decimal():
    assert formato(0.8126) == "0,813"
    assert formato(0.5, 1) == "0,5"
    assert formato(20.0, 1) == "20,0"


def test_formato_de_lo_que_no_esta_definido():
    assert formato(None) == "—"
    assert formato(None, ausente="n/a") == "n/a"


def test_csv_deja_vacio_lo_que_no_esta_definido(tmp_path):
    ruta = tmp_path / "x.csv"
    escribir_csv(ruta, ["a", "b", "c"], [{"a": "ñ.jpg", "b": None, "c": 0.5}])
    assert ruta.read_text(encoding="utf-8").splitlines() == ["a,b,c", "ñ.jpg,,0.5"]


def test_dibujar_cajas_pinta_referencia_y_predicha_sin_tocar_la_original():
    imagen = np.full((100, 120, 3), 128, dtype=np.uint8)
    copia = imagen.copy()
    predicha = Detection(box=(60, 60, 100, 90), score=0.8)
    salida = dibujar_cajas(imagen, [(10, 10, 50, 50)], [predicha])
    np.testing.assert_array_equal(imagen, copia)  # la original no cambia
    assert salida.shape == imagen.shape
    assert tuple(salida[10, 30]) == COLOR_REFERENCIA  # borde superior de la caja de referencia
    assert tuple(salida[60, 80]) == COLOR_PREDICHA  # borde superior de la predicha
    assert COLOR_REFERENCIA != COLOR_PREDICHA


def test_dibujar_cajas_sin_nada_devuelve_la_imagen_igual():
    imagen = np.full((50, 50, 3), 7, dtype=np.uint8)
    np.testing.assert_array_equal(dibujar_cajas(imagen, [], []), imagen)


def test_dibujar_cajas_recorta_las_que_se_salen():
    imagen = np.zeros((50, 50, 3), dtype=np.uint8)
    dibujar_cajas(imagen, [(-10, -10, 200, 200)], [Detection(box=(40, 40, 90, 90), score=0.5)])
