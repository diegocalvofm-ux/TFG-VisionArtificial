"""Pruebas de precisión, recall, F1, agregados, AP con pycocotools y percentiles."""

import numpy as np
import pytest

from src.eval.metricas import (
    ResultadoImagen,
    agregar,
    ap_coco,
    percentil,
    prf,
    resumen_latencia,
)


def test_prf_con_numeros_conocidos():
    m = prf(tp=3, fp=1, fn=2)
    assert m["precision"] == pytest.approx(0.75)
    assert m["recall"] == pytest.approx(0.6)
    assert m["f1"] == pytest.approx(2 / 3)


def test_prf_todo_correcto():
    m = prf(tp=5, fp=0, fn=0)
    assert (m["precision"], m["recall"], m["f1"]) == (1.0, 1.0, 1.0)


def test_prf_sin_predicciones_deja_la_precision_sin_definir():
    m = prf(tp=0, fp=0, fn=4)
    assert m["precision"] is None
    assert m["recall"] == 0.0
    assert m["f1"] == 0.0


def test_prf_sin_objetos_deja_el_recall_sin_definir():
    m = prf(tp=0, fp=2, fn=0)
    assert m["precision"] == 0.0
    assert m["recall"] is None
    assert m["f1"] == 0.0


def test_prf_sin_nada_no_esta_definido():
    assert prf(0, 0, 0) == {"precision": None, "recall": None, "f1": None}


def _resultado(tp, fp, fn, ious=(), n_pred=None):
    return ResultadoImagen(
        nombre="x.jpg", n_gt=tp + fn, n_pred=tp + fp if n_pred is None else n_pred,
        tp=tp, fp=fp, fn=fn, ious=list(ious), latencia_ms=1.0,
    )  # fmt: skip


def test_agregar_suma_los_conteos_y_promedia_todos_los_pares():
    resultados = [
        _resultado(2, 1, 0, ious=[1.0, 0.8]),
        _resultado(0, 0, 1),  # sin ninguna predicción
        _resultado(0, 1, 1),
    ]
    g = agregar(resultados)
    assert (g["tp"], g["fp"], g["fn"]) == (2, 2, 2)
    assert g["precision"] == pytest.approx(0.5)
    assert g["recall"] == pytest.approx(0.5)
    assert g["f1"] == pytest.approx(0.5)
    assert g["iou_medio"] == pytest.approx(0.9)
    assert g["imagenes"] == 3
    assert g["imagenes_sin_deteccion"] == 1


def test_agregar_sin_pares_deja_el_iou_medio_sin_definir():
    assert agregar([_resultado(0, 1, 1)])["iou_medio"] is None


def test_agregar_lista_vacia_es_un_error():
    with pytest.raises(ValueError, match="vací"):
        agregar([])


def test_imagen_sin_deteccion_es_la_que_no_tiene_predicciones():
    assert _resultado(0, 0, 3).sin_deteccion is True
    assert _resultado(0, 1, 3).sin_deteccion is False


def test_resultado_imagen_calcula_sus_metricas():
    r = _resultado(1, 1, 0, ious=[0.6, 0.8])
    assert r.iou_medio == pytest.approx(0.7)
    assert r.precision == pytest.approx(0.5)
    assert r.recall == pytest.approx(1.0)


# --- AP con pycocotools -----------------------------------------------------------------


def _gts():
    return [np.array([[0, 0, 100, 100]], dtype=float), np.array([[10, 10, 60, 60]], dtype=float)]


def test_ap_prediccion_perfecta():
    pycocotools = pytest.importorskip("pycocotools")
    assert pycocotools is not None
    preds = [(g.copy(), np.array([0.9])) for g in _gts()]
    ap50, ap50_95 = ap_coco(_gts(), preds, [(200, 200)] * 2)
    assert ap50 == pytest.approx(1.0)
    assert ap50_95 == pytest.approx(1.0)


def test_ap_sin_predicciones_es_cero():
    pytest.importorskip("pycocotools")
    vacias = [(np.zeros((0, 4)), np.zeros(0))] * 2
    assert ap_coco(_gts(), vacias, [(200, 200)] * 2) == (0.0, 0.0)


def test_ap_con_iou_intermedio_separa_ap50_de_ap50_95():
    pytest.importorskip("pycocotools")
    # Caja 100x100 frente a 60x100 desplazada: IoU = 0,6 con la de referencia.
    gts = [np.array([[0, 0, 100, 100]], dtype=float)]
    preds = [(np.array([[0, 0, 60, 100]], dtype=float), np.array([0.9]))]
    ap50, ap50_95 = ap_coco(gts, preds, [(200, 200)])
    assert ap50 == pytest.approx(1.0)
    assert 0.0 < ap50_95 < 1.0


def test_ap_sin_objetos_de_referencia_no_esta_definido():
    pytest.importorskip("pycocotools")
    preds = [(np.array([[0, 0, 10, 10]], dtype=float), np.array([0.9]))]
    assert ap_coco([np.zeros((0, 4))], preds, [(50, 50)]) == (None, None)


def test_ap_no_imprime_nada(capsys):
    pytest.importorskip("pycocotools")
    preds = [(g.copy(), np.array([0.9])) for g in _gts()]
    ap_coco(_gts(), preds, [(200, 200)] * 2)
    assert capsys.readouterr().out == ""


# --- Latencias ---------------------------------------------------------------------------


def test_percentil_con_valores_conocidos():
    assert percentil([10, 20, 30], 90) == pytest.approx(28.0)
    assert percentil([5], 90) == 5.0


def test_percentil_lista_vacia_es_un_error():
    with pytest.raises(ValueError, match="vací"):
        percentil([], 90)


def test_resumen_latencia():
    r = resumen_latencia([10.0, 20.0, 30.0])
    assert r == {"mediana_ms": 20.0, "p90_ms": pytest.approx(28.0)}
