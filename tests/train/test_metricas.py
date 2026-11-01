"""Pruebas de src/train/metricas.py: lectura del metrics.csv de Lightning y mejor punto."""

import pytest

from src.train.metricas import leer_metricas_csv, mejor_punto

CABECERA = "epoch,step,train/loss,val/mAP_50,val/mAP_50_95,val/mAR,val/F1,val/ema_mAP_50_95"

# Como el CSVLogger de Lightning: una fila por paso de entrenamiento (solo con train/*) y una
# fila de validación por época (solo con val/*), con las celdas que no tocan en blanco.
CSV_TIPICO = f"""{CABECERA}
0,9,0.90,,,,,
0,19,0.80,,,,,
0,19,,0.50,0.30,0.40,0.45,0.31
1,29,0.70,,,,,
1,39,0.60,,,,,
1,39,,0.70,0.45,0.55,0.60,0.46
2,49,0.50,,,,,
2,59,0.40,,,,,
2,59,,0.65,0.40,0.52,0.58,0.42
"""


def _csv(tmp_path, texto):
    ruta = tmp_path / "metrics.csv"
    ruta.write_text(texto, encoding="utf-8")
    return ruta


def test_una_fila_por_epoca_con_ap50_y_ap50_95(tmp_path):
    metricas = leer_metricas_csv(_csv(tmp_path, CSV_TIPICO))

    assert [m["epoca"] for m in metricas] == [1, 2, 3]  # Lightning cuenta desde 0; se guarda +1
    assert [m["epoca_csv"] for m in metricas] == [0, 1, 2]
    assert [m["AP50"] for m in metricas] == [0.50, 0.70, 0.65]
    assert [m["AP50_95"] for m in metricas] == [0.30, 0.45, 0.40]


def test_se_guardan_todas_las_metricas_de_valid_y_no_las_de_train(tmp_path):
    metricas = leer_metricas_csv(_csv(tmp_path, CSV_TIPICO))

    assert metricas[1]["val"] == {
        "val/mAP_50": 0.70,
        "val/mAP_50_95": 0.45,
        "val/mAR": 0.55,
        "val/F1": 0.60,
        "val/ema_mAP_50_95": 0.46,
    }
    assert "train/loss" not in metricas[1]["val"]


def test_las_celdas_vacias_o_nan_son_none_y_no_se_inventan(tmp_path):
    texto = f"{CABECERA}\n0,9,0.9,0.5,nan,,,\n1,19,0.8,,0.45,,,\n"

    metricas = leer_metricas_csv(_csv(tmp_path, texto))

    assert metricas[0]["AP50"] == 0.5 and metricas[0]["AP50_95"] is None
    assert metricas[1]["AP50"] is None and metricas[1]["AP50_95"] == 0.45
    assert "val/F1" not in metricas[0]["val"]


def test_si_una_epoca_repite_una_metrica_se_queda_con_el_ultimo_valor(tmp_path):
    texto = f"{CABECERA}\n0,5,,0.2,0.1,,,\n0,9,,0.5,0.3,,,\n"

    metricas = leer_metricas_csv(_csv(tmp_path, texto))

    assert len(metricas) == 1 and metricas[0]["AP50"] == 0.5


def test_las_epocas_se_devuelven_ordenadas(tmp_path):
    texto = f"{CABECERA}\n2,9,,0.9,0.8,,,\n0,9,,0.1,0.1,,,\n1,9,,0.5,0.4,,,\n"

    assert [m["epoca"] for m in leer_metricas_csv(_csv(tmp_path, texto))] == [1, 2, 3]


def test_un_csv_sin_columna_epoch_da_error(tmp_path):
    with pytest.raises(ValueError, match="epoch"):
        leer_metricas_csv(_csv(tmp_path, "step,val/mAP_50\n1,0.5\n"))


def test_un_csv_sin_ninguna_metrica_de_valid_da_error_claro(tmp_path):
    texto = "epoch,step,train/loss\n0,9,0.9\n1,19,0.8\n"

    with pytest.raises(ValueError, match="métricas de valid"):
        leer_metricas_csv(_csv(tmp_path, texto))


def test_un_csv_inexistente_da_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="metrics.csv"):
        leer_metricas_csv(tmp_path / "metrics.csv")


def test_el_mejor_punto_es_la_epoca_con_mayor_ap50_95(tmp_path):
    metricas = leer_metricas_csv(_csv(tmp_path, CSV_TIPICO))

    assert mejor_punto(metricas) == {
        "epoca": 2, "epoca_csv": 1, "AP50": 0.70, "AP50_95": 0.45
    }  # fmt: skip


def test_en_un_empate_gana_la_primera_epoca():
    metricas = [
        {"epoca": 1, "epoca_csv": 0, "AP50": 0.5, "AP50_95": 0.4, "val": {}},
        {"epoca": 2, "epoca_csv": 1, "AP50": 0.6, "AP50_95": 0.4, "val": {}},
    ]

    assert mejor_punto(metricas)["epoca"] == 1


def test_sin_ap50_95_no_hay_mejor_punto():
    assert mejor_punto([]) is None
    assert (
        mejor_punto([{"epoca": 1, "epoca_csv": 0, "AP50": 0.5, "AP50_95": None, "val": {}}]) is None
    )
