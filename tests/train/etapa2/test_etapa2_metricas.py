"""Pruebas de src/train/etapa2/metricas.py: métricas por grupo solo con las muestras visibles.

Los objetivos y las predicciones son índices de etiqueta (posición en ``GROUPS[grupo]``) y −1
para NO_VISIBLE, como los produce ``codificar``. Sin torch.
"""

import numpy as np
import pytest

from src.data.schema import GROUPS
from src.train.etapa2.metricas import (
    clases_mayoritarias,
    criterio,
    evaluar_conjunto,
    exactitud_equilibrada,
    f1_macro,
    matriz_confusion,
    metricas_grupo,
    separar_matrices,
)

GRUPOS = list(GROUPS)
BIO = GRUPOS.index("daño_bio")
CORTE = GRUPOS.index("corte")


def _tabla(filas: list[dict]) -> np.ndarray:
    """Matriz (N, 6) con −1 salvo en los grupos que da cada fila (``{grupo: etiqueta}``)."""
    salida = np.full((len(filas), len(GRUPOS)), -1, dtype=np.int64)
    for i, fila in enumerate(filas):
        for grupo, etiqueta in fila.items():
            salida[i, GRUPOS.index(grupo)] = GROUPS[grupo].index(etiqueta)
    return salida


# --- métricas básicas ---------------------------------------------------------------------


def test_matriz_de_confusion_filas_referencia_y_columnas_prediccion():
    matriz = matriz_confusion(np.array([0, 0, 1, 2]), np.array([0, 1, 1, 0]), 3)

    assert matriz.tolist() == [[1, 1, 0], [0, 1, 0], [1, 0, 0]]


def test_exactitud_equilibrada_con_numeros_conocidos():
    # recall de la clase 0: 2/3; de la clase 1: 1/1 -> media 5/6
    matriz = matriz_confusion(np.array([0, 0, 0, 1]), np.array([0, 0, 1, 1]), 2)

    assert exactitud_equilibrada(matriz) == pytest.approx(5 / 6)


def test_exactitud_equilibrada_solo_promedia_las_clases_con_referencia():
    # La clase 1 solo aparece predicha y la 2 nunca: solo cuenta el recall de la 0 (1/2).
    matriz = matriz_confusion(np.array([0, 0]), np.array([0, 1]), 3)

    assert exactitud_equilibrada(matriz) == pytest.approx(0.5)


def test_f1_macro_con_numeros_conocidos():
    # clase 0: tp 2, fp 0, fn 1 -> 4/5; clase 1: tp 1, fp 1, fn 0 -> 2/3
    matriz = matriz_confusion(np.array([0, 0, 0, 1]), np.array([0, 0, 1, 1]), 2)

    assert f1_macro(matriz) == pytest.approx((4 / 5 + 2 / 3) / 2)


def test_f1_macro_incluye_las_clases_predichas_sin_referencia():
    # clase 0: tp 1, fn 1 -> 2/3; clase 1 predicha sin referencia -> 0; clase 2 no aparece
    matriz = matriz_confusion(np.array([0, 0]), np.array([0, 1]), 3)

    assert f1_macro(matriz) == pytest.approx((2 / 3 + 0) / 2)


def test_sin_muestras_las_metricas_no_estan_definidas():
    vacia = matriz_confusion(np.array([], dtype=int), np.array([], dtype=int), 3)

    assert exactitud_equilibrada(vacia) is None and f1_macro(vacia) is None


# --- por grupo, enmascarado ---------------------------------------------------------------


def test_metricas_grupo_ignora_las_filas_no_visibles():
    resultado = metricas_grupo(np.array([0, -1, 1, -1]), np.array([0, 1, 1, 0]), ("A", "B"))

    assert resultado["n"] == 2
    assert resultado["exactitud_equilibrada"] == 1.0 and resultado["f1_macro"] == 1.0
    assert resultado["matriz"] == [[1, 0], [0, 1]]
    assert resultado["etiquetas"] == ["A", "B"] and resultado["soporte"] == {"A": 1, "B": 1}


def test_un_grupo_sin_muestras_visibles_da_n_0_y_metricas_vacias():
    resultado = metricas_grupo(np.array([-1, -1]), np.array([0, 1]), ("A", "B"))

    assert resultado["n"] == 0
    assert resultado["exactitud_equilibrada"] is None and resultado["f1_macro"] is None
    assert resultado["matriz"] == [[0, 0], [0, 0]]


def test_clases_mayoritarias_por_grupo():
    objetivos = _tabla([
        {"daño_bio": "ROT_DECAY", "corte": "CUT_OK"},
        {"daño_bio": "ROT_DECAY", "corte": "ROOT_REMAINS"},
        {"daño_bio": "BIO_OK"},
    ])  # fmt: skip

    mayoritarias = clases_mayoritarias(objetivos)

    assert mayoritarias["daño_bio"] == "ROT_DECAY"
    assert mayoritarias["corte"] == "CUT_OK"  # empate 1-1: gana el orden de schema
    assert mayoritarias["color"] is None  # sin muestras visibles en train


# --- conjunto completo --------------------------------------------------------------------


@pytest.fixture
def evaluacion():
    objetivos = _tabla([
        {"daño_bio": "BIO_OK", "corte": "CUT_OK"},
        {"daño_bio": "BIO_OK", "corte": "STEM_TOO_LONG"},
        {"daño_bio": "ROT_DECAY"},
        {"daño_bio": "ROT_DECAY"},
    ])  # fmt: skip
    predicciones = _tabla([
        {"daño_bio": "BIO_OK", "corte": "CUT_OK", "color": "WILTING"},
        {"daño_bio": "ROT_DECAY", "corte": "CUT_OK", "color": "WILTING"},
        {"daño_bio": "ROT_DECAY", "corte": "CUT_OK"},
        {"daño_bio": "ROT_DECAY", "corte": "CUT_OK"},
    ])  # fmt: skip
    fuentes = ["propio", "propio", "phs", "phs"]
    mayoritarias = {g: None for g in GROUPS} | {"daño_bio": "BIO_OK", "corte": "CUT_OK"}
    return evaluar_conjunto(objetivos, predicciones, fuentes, mayoritarias)


def test_evaluar_conjunto_mide_cada_grupo_solo_con_sus_visibles(evaluacion):
    bio = evaluacion["grupos"]["daño_bio"]["modelo"]
    assert bio["n"] == 4
    assert bio["exactitud_equilibrada"] == pytest.approx((1 / 2 + 1) / 2)
    corte = evaluacion["grupos"]["corte"]["modelo"]
    assert corte["n"] == 2 and corte["exactitud_equilibrada"] == pytest.approx(0.5)
    assert evaluacion["grupos"]["color"]["modelo"]["n"] == 0  # las predicciones no cuentan


def test_evaluar_conjunto_compara_con_el_clasificador_mayoritario(evaluacion):
    bio = evaluacion["grupos"]["daño_bio"]["mayoritaria"]
    # Siempre BIO_OK: recall 1 en BIO_OK y 0 en ROT_DECAY.
    assert bio["clase"] == "BIO_OK"
    assert bio["exactitud_equilibrada"] == pytest.approx(0.5)
    assert bio["f1_macro"] == pytest.approx((2 / 3 + 0) / 2)
    color = evaluacion["grupos"]["color"]["mayoritaria"]
    assert color == {"clase": None, "exactitud_equilibrada": None, "f1_macro": None}


def test_evaluar_conjunto_desglosa_por_fuente(evaluacion):
    por_fuente = evaluacion["por_fuente"]

    assert list(por_fuente) == ["propio", "phs"]
    assert por_fuente["phs"]["daño_bio"]["modelo"]["n"] == 2
    assert por_fuente["phs"]["daño_bio"]["modelo"]["exactitud_equilibrada"] == 1.0
    assert por_fuente["phs"]["corte"]["modelo"]["n"] == 0
    assert por_fuente["propio"]["daño_bio"]["modelo"]["exactitud_equilibrada"] == 0.5


def test_el_criterio_es_la_media_de_los_grupos_con_datos(evaluacion):
    assert evaluacion["criterio"] == pytest.approx((0.75 + 0.5) / 2)
    assert criterio(evaluacion["grupos"]) == evaluacion["criterio"]
    assert evaluacion["grupos_con_datos"] == ["corte", "daño_bio"]


def test_sin_ningun_grupo_con_datos_el_criterio_es_none():
    vacio = np.full((2, len(GRUPOS)), -1, dtype=np.int64)

    resultado = evaluar_conjunto(vacio, vacio, ["a", "a"], dict.fromkeys(GROUPS))

    assert resultado["criterio"] is None and resultado["grupos_con_datos"] == []


def test_las_formas_distintas_dan_error():
    with pytest.raises(ValueError, match="forma"):
        evaluar_conjunto(np.zeros((2, 6), dtype=int), np.zeros((3, 6), dtype=int), ["a"] * 2,
                         dict.fromkeys(GROUPS))  # fmt: skip


def test_separar_matrices_las_quita_del_resumen_y_las_junta_aparte(evaluacion):
    sin, matrices = separar_matrices(evaluacion)

    assert "matriz" not in str(sin)
    assert matrices["grupos"]["daño_bio"]["matriz"] == evaluacion["grupos"]["daño_bio"]["modelo"][
        "matriz"
    ]
    assert matrices["grupos"]["daño_bio"]["etiquetas"] == list(GROUPS["daño_bio"])
    assert matrices["por_fuente"]["phs"]["daño_bio"]["matriz"][1][1] == 2
    assert "matriz" in evaluacion["grupos"]["daño_bio"]["modelo"]  # el original no cambia
