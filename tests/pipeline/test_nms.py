"""Pruebas de src/pipeline/nms.py: supresión de cajas solapadas (NMS por IoU)."""

import numpy as np
import pytest

from src.pipeline.nms import nms


def _cajas(*cajas):
    return np.array(cajas, dtype=float)


def _puntuaciones(*valores):
    return np.array(valores, dtype=float)


def test_caso4_entre_dos_cajas_muy_solapadas_se_queda_la_de_mayor_puntuacion():
    cajas = _cajas((0, 0, 100, 100), (5, 5, 105, 105))  # IoU = 0.82

    assert nms(cajas, _puntuaciones(0.6, 0.9), 0.5) == [1]
    assert nms(cajas, _puntuaciones(0.9, 0.6), 0.5) == [0]


def test_caso5_dos_cajas_que_no_se_solapan_se_conservan_las_dos():
    cajas = _cajas((0, 0, 10, 10), (50, 50, 60, 60))

    assert nms(cajas, _puntuaciones(0.5, 0.9), 0.5) == [1, 0]


def test_cajas_identicas_dejan_solo_la_mejor():
    cajas = _cajas((0, 0, 10, 10), (0, 0, 10, 10), (0, 0, 10, 10))

    assert nms(cajas, _puntuaciones(0.2, 0.8, 0.5), 0.5) == [1]


def test_cajas_que_solo_se_tocan_en_un_borde_no_se_suprimen():
    cajas = _cajas((0, 0, 10, 10), (10, 0, 20, 10))  # IoU = 0

    assert nms(cajas, _puntuaciones(0.9, 0.8), 0.0) == [0, 1]


def test_es_voraz_una_caja_suprimida_no_suprime_a_otras():
    # B solapa con A y con C, pero A no con C. Se queda A, se suprime B y, como B ya no
    # cuenta, C se conserva.
    a, b, c = (0, 0, 10, 10), (6, 0, 16, 10), (12, 0, 22, 10)  # IoU(A,B) = IoU(B,C) = 0.25

    assert nms(_cajas(a, b, c), _puntuaciones(0.9, 0.8, 0.7), 0.2) == [0, 2]


def test_con_iou_justo_en_el_umbral_se_conservan_las_dos():
    cajas = _cajas((0, 0, 10, 10), (0, 0, 10, 5))  # IoU = 0.5 exacto

    assert nms(cajas, _puntuaciones(0.9, 0.8), 0.5) == [0, 1]  # se suprime solo si IoU > umbral
    assert nms(cajas, _puntuaciones(0.9, 0.8), 0.49) == [0]


def test_el_resultado_va_de_mayor_a_menor_puntuacion():
    cajas = _cajas((0, 0, 5, 5), (20, 0, 25, 5), (40, 0, 45, 5))

    assert nms(cajas, _puntuaciones(0.2, 0.9, 0.5), 0.5) == [1, 2, 0]


def test_con_empate_de_puntuacion_manda_el_indice_menor():
    cajas = _cajas((0, 0, 5, 5), (20, 0, 25, 5))

    assert nms(cajas, _puntuaciones(0.5, 0.5), 0.5) == [0, 1]


def test_devuelve_enteros_de_python():
    resultado = nms(_cajas((0, 0, 5, 5)), _puntuaciones(0.5), 0.5)

    assert resultado == [0] and all(type(i) is int for i in resultado)


def test_entrada_vacia_devuelve_lista_vacia():
    assert nms(np.zeros((0, 4)), np.zeros(0), 0.5) == []


def test_cajas_de_area_cero_no_dan_error():
    cajas = _cajas((5, 5, 5, 5), (5, 5, 5, 5))  # unión 0: el IoU se toma como 0

    assert nms(cajas, _puntuaciones(0.9, 0.8), 0.5) == [0, 1]


@pytest.mark.parametrize(
    ("cajas", "puntuaciones", "umbral"),
    [
        (np.zeros((2, 3)), np.zeros(2), 0.5),  # las cajas deben tener 4 columnas
        (np.zeros((2, 4)), np.zeros(3), 0.5),  # una puntuación por caja
        (np.zeros(4), np.zeros(1), 0.5),  # las cajas deben ser una matriz
        (np.zeros((2, 4)), np.zeros((2, 1)), 0.5),  # las puntuaciones, un vector
        (np.zeros((1, 4)), np.zeros(1), -0.1),  # umbral fuera de [0, 1]
        (np.zeros((1, 4)), np.zeros(1), 1.5),
    ],
    ids=["tres_columnas", "puntuaciones_de_mas", "cajas_1d", "puntuaciones_2d", "neg", "mayor_1"],
)
def test_entradas_incorrectas_lanzan_value_error(cajas, puntuaciones, umbral):
    with pytest.raises(ValueError):
        nms(cajas, puntuaciones, umbral)


@pytest.mark.parametrize("iou", [0.3, 0.5, 0.7])
def test_coincide_con_torchvision_en_cajas_aleatorias(iou):
    torch = pytest.importorskip("torch")
    ops = pytest.importorskip("torchvision.ops")
    azar = np.random.default_rng(0)
    xy = azar.uniform(0, 100, (200, 2))
    cajas = np.hstack([xy, xy + azar.uniform(1, 40, (200, 2))]).astype(np.float32)
    puntuaciones = azar.permutation(200).astype(np.float32) / 200  # todas distintas

    esperado = ops.nms(torch.from_numpy(cajas), torch.from_numpy(puntuaciones), iou).tolist()

    assert nms(cajas, puntuaciones, iou) == esperado
