"""Pruebas del IoU y del emparejamiento voraz por confianza (casos con la solución conocida)."""

import numpy as np
import pytest

from src.eval.cajas import emparejar, iou, matriz_iou


def test_iou_cajas_identicas():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)


def test_iou_cajas_disjuntas():
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_iou_cajas_que_solo_se_tocan_en_el_borde():
    assert iou((0, 0, 10, 10), (10, 0, 20, 10)) == 0.0


def test_iou_mitad_solapada():
    # Común 5x10 = 50; unión 100 + 100 - 50 = 150.
    assert iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)


def test_iou_caja_contenida():
    # Común = 25 (la pequeña); unión = 100.
    assert iou((0, 0, 10, 10), (2, 2, 7, 7)) == pytest.approx(0.25)


def test_iou_es_simetrico():
    a, b = (0, 0, 10, 10), (3, 4, 15, 12)
    assert iou(a, b) == pytest.approx(iou(b, a))


def test_iou_caja_sin_area_vale_cero():
    assert iou((5, 5, 5, 5), (5, 5, 5, 5)) == 0.0
    assert iou((0, 0, 10, 10), (5, 5, 5, 20)) == 0.0


def test_matriz_iou_forma_y_valores():
    matriz = matriz_iou([(0, 0, 10, 10), (5, 0, 15, 10)], [(0, 0, 10, 10)])
    assert matriz.shape == (2, 1)
    assert matriz[0, 0] == pytest.approx(1.0)
    assert matriz[1, 0] == pytest.approx(1 / 3)


def test_matriz_iou_vacia():
    assert matriz_iou([], [(0, 0, 1, 1)]).shape == (0, 1)
    assert matriz_iou([(0, 0, 1, 1)], []).shape == (1, 0)


def test_emparejamiento_perfecto():
    emp = emparejar([(0, 0, 10, 10), (20, 20, 30, 30)], [0.9, 0.8],
                    [(20, 20, 30, 30), (0, 0, 10, 10)], 0.5)  # fmt: skip
    assert sorted((p, g) for p, g, _ in emp.pares) == [(0, 1), (1, 0)]
    assert emp.fp == [] and emp.fn == []


def test_un_objeto_con_dos_predicciones_da_un_tp_y_un_fp():
    # Las dos cajas cubren el único objeto; solo la de mayor confianza se lo queda.
    emp = emparejar([(0, 0, 10, 10), (1, 1, 11, 11)], [0.6, 0.9], [(0, 0, 10, 10)], 0.5)
    assert [(p, g) for p, g, _ in emp.pares] == [(1, 0)]  # la predicción 1 (confianza 0,9)
    assert emp.fp == [0]
    assert emp.fn == []


def test_prediccion_sin_objeto_es_falso_positivo():
    emp = emparejar([(50, 50, 60, 60)], [0.9], [(0, 0, 10, 10)], 0.5)
    assert emp.pares == []
    assert emp.fp == [0]
    assert emp.fn == [0]


def test_objeto_sin_prediccion_es_falso_negativo():
    emp = emparejar([], [], [(0, 0, 10, 10), (20, 20, 30, 30)], 0.5)
    assert emp.pares == [] and emp.fp == [] and emp.fn == [0, 1]


def test_sin_objetos_todo_es_falso_positivo():
    emp = emparejar([(0, 0, 10, 10)], [0.5], [], 0.5)
    assert emp.fp == [0] and emp.fn == []


def test_orden_por_confianza_manda_sobre_el_mejor_iou():
    # La predicción 0 (confianza baja) encaja mejor con el objeto, pero la 1 llega antes y lo toma.
    gt = [(0, 0, 10, 10)]
    emp = emparejar([(0, 0, 10, 10), (1, 0, 11, 10)], [0.5, 0.9], gt, 0.5)
    assert [(p, g) for p, g, _ in emp.pares] == [(1, 0)]
    assert emp.fp == [0]


def test_la_segunda_prediccion_toma_otro_objeto_libre():
    # Dos predicciones muy solapadas y dos objetos solapados: cada una se queda uno distinto.
    gts = [(0, 0, 10, 10), (4, 0, 14, 10)]
    emp = emparejar([(0, 0, 10, 10), (4, 0, 14, 10)], [0.9, 0.8], gts, 0.5)
    assert sorted((p, g) for p, g, _ in emp.pares) == [(0, 0), (1, 1)]


def test_iou_exactamente_en_el_umbral_cuenta():
    # IoU = 1/3 con estas cajas; un umbral de 1/3 las empareja y uno algo mayor, no.
    pred, gt = [(0, 0, 10, 10)], [(5, 0, 15, 10)]
    assert emparejar(pred, [0.9], gt, 1 / 3).pares != []
    assert emparejar(pred, [0.9], gt, 0.34).pares == []


def test_el_iou_de_cada_par_se_devuelve():
    emp = emparejar([(0, 0, 10, 10)], [0.9], [(0, 0, 10, 5)], 0.5)
    assert emp.pares[0][2] == pytest.approx(0.5)


def test_empate_de_confianza_gana_el_primero():
    emp = emparejar([(0, 0, 10, 10), (0, 0, 10, 10)], [0.7, 0.7], [(0, 0, 10, 10)], 0.5)
    assert [(p, g) for p, g, _ in emp.pares] == [(0, 0)]
    assert emp.fp == [1]


def test_longitudes_distintas_son_un_error():
    with pytest.raises(ValueError, match="confianza"):
        emparejar([(0, 0, 1, 1)], [0.5, 0.6], [], 0.5)


def test_acepta_arrays_de_numpy():
    emp = emparejar(np.array([[0, 0, 10, 10]]), np.array([0.9]), np.array([[0, 0, 10, 10]]), 0.5)
    assert len(emp.pares) == 1
