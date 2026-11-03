"""Pruebas de la línea base HSV con imágenes sintéticas (círculos verdes sobre fondo gris)."""

import cv2
import numpy as np
import pytest

from src.eval.cajas import iou
from src.eval.config import ParametrosHSV
from src.eval.linea_base import LineaBaseHSV

# Mismos valores que configs/eval/localizacion.yaml (la prueba de configuración comprueba el YAML).
PARAMETROS = ParametrosHSV(
    h=(35, 85), s=(40, 255), v=(40, 255), apertura_px=5, cierre_px=15, area_min_frac=0.005
)

VERDE_BGR = (0, 200, 0)
GRIS_BGR = (128, 128, 128)


def _fondo(alto=200, ancho=300):
    return np.full((alto, ancho, 3), GRIS_BGR, dtype=np.uint8)


def _con_circulo(imagen, centro, radio):
    cv2.circle(imagen, centro, radio, VERDE_BGR, thickness=-1)
    return imagen


def test_circulo_verde_sobre_gris_da_una_caja_correcta():
    imagen = _con_circulo(_fondo(), (150, 100), 40)
    detecciones = LineaBaseHSV(PARAMETROS).detect(imagen)
    assert len(detecciones) == 1
    esperada = (110, 60, 191, 141)  # el círculo ocupa x e y desde centro-radio hasta centro+radio
    assert iou(detecciones[0].box, esperada) >= 0.9
    assert detecciones[0].score == 1.0


def test_fondo_gris_sin_verde_no_da_cajas():
    assert LineaBaseHSV(PARAMETROS).detect(_fondo()) == []


def test_una_mota_pequena_se_descarta_por_area_minima():
    # Área mínima = 0,005 * 200 * 300 = 300 px; un círculo de radio 8 tiene ~200 px.
    imagen = _con_circulo(_fondo(), (50, 50), 8)
    assert LineaBaseHSV(PARAMETROS).detect(imagen) == []


def test_dos_circulos_separados_dan_dos_cajas_ordenadas_por_area():
    imagen = _fondo()
    _con_circulo(imagen, (60, 100), 25)
    _con_circulo(imagen, (200, 100), 40)
    detecciones = LineaBaseHSV(PARAMETROS).detect(imagen)
    assert len(detecciones) == 2
    anchos = [d.box[2] - d.box[0] for d in detecciones]
    assert anchos[0] > anchos[1]  # primero el más grande


def test_dos_circulos_que_se_tocan_se_funden_en_una_componente():
    # Limitación conocida de la línea base: lechugas pegadas salen como un único objeto.
    imagen = _fondo()
    _con_circulo(imagen, (100, 100), 30)
    _con_circulo(imagen, (150, 100), 30)
    assert len(LineaBaseHSV(PARAMETROS).detect(imagen)) == 1


def test_la_caja_queda_dentro_de_la_imagen_y_con_enteros():
    imagen = _con_circulo(_fondo(), (290, 190), 40)  # se sale por la esquina
    caja = LineaBaseHSV(PARAMETROS).detect(imagen)[0].box
    assert all(isinstance(v, int) for v in caja)
    assert 0 <= caja[0] < caja[2] <= 300 and 0 <= caja[1] < caja[3] <= 200


def test_el_rojo_no_se_detecta_como_verde():
    imagen = _fondo()
    cv2.circle(imagen, (150, 100), 40, (0, 0, 200), thickness=-1)
    assert LineaBaseHSV(PARAMETROS).detect(imagen) == []


@pytest.mark.parametrize("campo", ["apertura_px", "cierre_px"])
def test_un_kernel_de_un_pixel_no_cambia_la_mascara(campo):
    parametros = ParametrosHSV(**{**PARAMETROS.__dict__, campo: 1})
    imagen = _con_circulo(_fondo(), (150, 100), 40)
    assert len(LineaBaseHSV(parametros).detect(imagen)) == 1
