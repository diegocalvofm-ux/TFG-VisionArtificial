"""Pruebas de src/train/etapa2/datos.py: lectura de recortes.csv, reparto, smoke y sobremuestreo.

Todo con el dataset sintético del conftest (sin torch, sin imágenes reales).
"""

from collections import Counter

import numpy as np
import pandas as pd
import pytest

from src.data.schema import GROUPS, NO_VISIBLE
from src.train.etapa2.datos import (
    codificar,
    indices_por_epoca,
    leer_recortes,
    recuentos,
    repartir_propio,
    subconjunto,
)


@pytest.fixture
def tabla(dataset_sintetico) -> pd.DataFrame:
    df, _ = leer_recortes(dataset_sintetico())
    return df


# --- lectura ------------------------------------------------------------------------------


def test_leer_recortes_devuelve_todas_las_filas_con_su_imagen(dataset_sintetico):
    carpeta = dataset_sintetico()

    df, sin_etiquetas = leer_recortes(carpeta)

    assert len(df) == 12 + 16 + 16 and sin_etiquetas == []
    assert Counter(df["particion"]) == {"train": 28, "valid": 8, "test": 8}
    assert all(c in df.columns for c in GROUPS)
    assert list(df.index) == list(range(len(df)))  # índice limpio, posición = fila
    assert df.loc[0, "ruta"] == carpeta / "imagenes" / "propio_000001.jpg"


def test_sin_recortes_csv_se_explica_como_generarlo(tmp_path):
    with pytest.raises(FileNotFoundError, match="src.data.etapa2"):
        leer_recortes(tmp_path / "vacio")


def test_una_tabla_no_valida_da_un_error_con_el_problema(dataset_sintetico, fabricas):
    filas = fabricas.filas()
    filas[3]["corte"] = "MUY_CORTO"

    with pytest.raises(ValueError, match="recortes.csv") as error:
        leer_recortes(dataset_sintetico(filas=filas))

    assert "MUY_CORTO" in str(error.value) and "corte" in str(error.value)


def test_si_faltan_imagenes_el_error_dice_cuantas_y_cuales(dataset_sintetico):
    carpeta = dataset_sintetico()
    (carpeta / "imagenes" / "yssp_000002.jpg").unlink()
    (carpeta / "imagenes" / "phs_000005.jpg").unlink()

    with pytest.raises(FileNotFoundError, match="2 imágenes") as error:
        leer_recortes(carpeta)

    assert "yssp_000002.jpg" in str(error.value) and "phs_000005.jpg" in str(error.value)


def test_las_filas_sin_ningun_grupo_visible_se_quitan_y_se_cuentan(dataset_sintetico, fabricas):
    filas = fabricas.filas()
    for grupo in GROUPS:
        filas[0][grupo] = NO_VISIBLE

    df, sin_etiquetas = leer_recortes(dataset_sintetico(filas=filas))

    assert sin_etiquetas == ["propio_000001"]
    assert "propio_000001" not in set(df["recorte_id"]) and len(df) == 43
    assert list(df.index) == list(range(43))


def test_la_lectura_no_modifica_el_dataset(dataset_sintetico):
    carpeta = dataset_sintetico()
    antes = {p: p.read_bytes() for p in carpeta.rglob("*") if p.is_file()}

    leer_recortes(carpeta)

    assert {p: p.read_bytes() for p in carpeta.rglob("*") if p.is_file()} == antes


# --- codificación -------------------------------------------------------------------------


def test_codificar_pone_la_posicion_de_schema_y_menos_uno_en_no_visible(tabla):
    objetivos = codificar(tabla)

    assert objetivos.shape == (len(tabla), len(GROUPS)) and objetivos.dtype == np.int64
    corte = list(GROUPS).index("corte")
    bio = list(GROUPS).index("daño_bio")
    # propio_000002: corte = STEM_TOO_LONG; propio_000003: corte = NO_VISIBLE
    assert objetivos[1, corte] == GROUPS["corte"].index("STEM_TOO_LONG")
    assert objetivos[2, corte] == -1
    fila_phs = tabla.index[tabla["recorte_id"] == "phs_000001"][0]
    assert objetivos[fila_phs, bio] == GROUPS["daño_bio"].index("ROT_DECAY")
    assert objetivos[fila_phs, corte] == -1


# --- reparto de la fuente propia por lechuga_id ------------------------------------------


def test_reparto_propio_no_deja_ninguna_lechuga_en_dos_particiones(tabla):
    repartida, resumen = repartir_propio(tabla, "propio", (0.5, 0.25, 0.25), semilla=42)

    propias = repartida[repartida["fuente"] == "propio"]
    assert (propias.groupby("lechuga_id")["particion"].nunique() == 1).all()
    assert set(propias["particion"]) == {"train", "valid", "test"}
    assert sorted(sum(resumen["lechugas"].values(), [])) == [f"L{i}" for i in range(1, 7)]
    assert resumen["fracciones"] == {"train": 0.5, "valid": 0.25, "test": 0.25}


def test_reparto_propio_no_toca_las_otras_fuentes_ni_la_tabla_original(tabla):
    copia = tabla.copy()

    repartida, _ = repartir_propio(tabla, "propio", (0.5, 0.25, 0.25), semilla=42)

    pd.testing.assert_frame_equal(tabla, copia)
    publicas = repartida["fuente"] != "propio"
    assert (repartida.loc[publicas, "particion"] == tabla.loc[publicas, "particion"]).all()


def test_reparto_propio_es_determinista_con_la_semilla(tabla):
    a, _ = repartir_propio(tabla, "propio", (0.5, 0.25, 0.25), semilla=7)
    b, _ = repartir_propio(tabla, "propio", (0.5, 0.25, 0.25), semilla=7)

    assert list(a["particion"]) == list(b["particion"])


def test_reparto_propio_con_menos_de_tres_lechugas_da_un_error_claro(tabla):
    pocas = tabla[~tabla["lechuga_id"].isin(["L3", "L4", "L5", "L6"])].reset_index(drop=True)

    with pytest.raises(ValueError, match="lechuga") as error:
        repartir_propio(pocas, "propio", (0.7, 0.15, 0.15), semilla=42)

    assert "propio" in str(error.value)


def test_reparto_de_una_fuente_sin_filas_da_error(tabla):
    with pytest.raises(ValueError, match="'otra'"):
        repartir_propio(tabla, "otra", (0.7, 0.15, 0.15), semilla=42)


# --- sobremuestreo -------------------------------------------------------------------------


def test_el_sobremuestreo_repite_solo_las_filas_propias_de_train(tabla):
    train = tabla[tabla["particion"] == "train"].reset_index(drop=True)

    indices = indices_por_epoca(train, "propio", 3)

    veces = Counter(indices)
    for posicion, fuente in enumerate(train["fuente"]):
        assert veces[posicion] == (3 if fuente == "propio" else 1)
    assert len(indices) == 16 + 12 * 3


def test_con_factor_1_no_se_repite_nada(tabla):
    train = tabla[tabla["particion"] == "train"].reset_index(drop=True)

    assert indices_por_epoca(train, "propio", 1) == list(range(len(train)))


def test_el_sobremuestreo_no_admite_filas_de_valid_ni_de_test(tabla):
    with pytest.raises(ValueError, match="solo train"):
        indices_por_epoca(tabla, "propio", 2)


def test_las_filas_propias_que_van_a_valid_o_test_no_se_repiten(tabla):
    repartida, _ = repartir_propio(tabla, "propio", (0.5, 0.25, 0.25), semilla=42)
    train = repartida[repartida["particion"] == "train"].reset_index(drop=True)

    indices = indices_por_epoca(train, "propio", 4)

    propias_train = int((train["fuente"] == "propio").sum())
    assert 0 < propias_train < 12
    assert len(indices) == 16 + propias_train * 4  # las de valid y test no están


@pytest.mark.parametrize("factor", [0, -1, 1.5, True])
def test_un_factor_no_valido_da_error(tabla, factor):
    train = tabla[tabla["particion"] == "train"].reset_index(drop=True)

    with pytest.raises(ValueError, match="sobremuestreo"):
        indices_por_epoca(train, "propio", factor)


# --- subconjunto del smoke ----------------------------------------------------------------


def test_el_subconjunto_del_smoke_tiene_n_filas_de_todas_las_fuentes(tabla):
    train = tabla[tabla["particion"] == "train"].reset_index(drop=True)

    sub = subconjunto(train, 6, semilla=42)

    assert len(sub) == 6 and set(sub["fuente"]) == {"propio", "yssp", "phs"}
    assert list(sub.index) == list(range(6))
    assert set(sub["recorte_id"]) <= set(train["recorte_id"])


def test_el_subconjunto_garantiza_la_fuente_pequena(tabla):
    # 1 recorte propio entre 17: proporcionalmente le tocaría 0,35 de 6, pero entra igual.
    train = tabla[tabla["particion"] == "train"]
    un_propio = pd.concat([train[train["fuente"] == "propio"].head(1),
                           train[train["fuente"] != "propio"]]).reset_index(drop=True)  # fmt: skip

    sub = subconjunto(un_propio, 6, semilla=42)

    assert (sub["fuente"] == "propio").sum() == 1


def test_el_subconjunto_es_determinista_con_la_semilla(tabla):
    a = subconjunto(tabla, 10, semilla=3)
    b = subconjunto(tabla, 10, semilla=3)

    assert list(a["recorte_id"]) == list(b["recorte_id"])


def test_si_n_no_es_menor_que_el_total_se_devuelve_todo(tabla):
    valid = tabla[tabla["particion"] == "valid"].reset_index(drop=True)

    pd.testing.assert_frame_equal(subconjunto(valid, 100, semilla=42), valid)


# --- recuentos ----------------------------------------------------------------------------


def test_recuentos_por_particion_fuente_grupo_y_etiqueta(tabla):
    cuentas = recuentos(tabla)

    propio = cuentas["train"]["propio"]
    assert propio["recortes"] == 12
    assert propio["etiquetas"]["corte"] == {
        "CUT_OK": 4, "STEM_TOO_LONG": 4, "UNCLEAN_CUT": 0, "ROOT_REMAINS": 0, NO_VISIBLE: 4
    }  # fmt: skip
    assert cuentas["valid"]["yssp"]["recortes"] == 4
    assert cuentas["test"]["phs"]["etiquetas"]["corte"][NO_VISIBLE] == 4
    assert "propio" not in cuentas["valid"]  # una fuente sin filas en una partición no aparece
