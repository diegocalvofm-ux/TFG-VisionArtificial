"""Pruebas de src/data/split.py: partición train/val/test por lechuga."""

import warnings

import pandas as pd
import pytest

from src.data.split import group_split

PARTICIONES = {"train", "val", "test"}


def _particion_por_lechuga(df: pd.DataFrame) -> dict:
    """Devuelve {lechuga_id: particion} (falla si una lechuga tiene dos particiones)."""
    return df.groupby("lechuga_id")["split"].agg(lambda s: set(s)).to_dict()


# --- Caso 1: ninguna lechuga en dos particiones ----------------------------------------


def test_ninguna_lechuga_aparece_en_dos_particiones(metadata_valida):
    df = metadata_valida(n_lechugas=30, fotos_por_lechuga=4)

    salida = group_split(df)

    por_lechuga = _particion_por_lechuga(salida)
    assert all(len(particiones) == 1 for particiones in por_lechuga.values())
    assert set(salida["split"]) == PARTICIONES


# --- Caso 2: determinismo y semilla ----------------------------------------------------


def test_misma_semilla_mismo_resultado(metadata_valida):
    df = metadata_valida(n_lechugas=100, fotos_por_lechuga=2)

    pd.testing.assert_frame_equal(group_split(df, seed=42), group_split(df, seed=42))


def test_otra_semilla_otro_resultado(metadata_valida):
    df = metadata_valida(n_lechugas=100, fotos_por_lechuga=2)

    a = group_split(df, seed=42)["split"]
    b = group_split(df, seed=43)["split"]

    assert not a.equals(b)


def test_el_resultado_no_depende_del_orden_de_las_filas(metadata_valida):
    df = metadata_valida(n_lechugas=40, fotos_por_lechuga=3)
    barajado = df.sample(frac=1, random_state=0)

    a = _particion_por_lechuga(group_split(df))
    b = _particion_por_lechuga(group_split(barajado))

    assert a == b


# --- Caso 3: proporciones --------------------------------------------------------------


def test_con_100_grupos_las_proporciones_se_parecen(metadata_valida):
    df = metadata_valida(n_lechugas=100, fotos_por_lechuga=3)

    salida = group_split(df, fracciones=(0.7, 0.15, 0.15))

    lechugas = salida.drop_duplicates("lechuga_id")["split"].value_counts()
    assert abs(lechugas["train"] - 70) <= 1
    assert abs(lechugas["val"] - 15) <= 1
    assert abs(lechugas["test"] - 15) <= 1
    assert lechugas.sum() == 100


# --- Caso 4: errores de entrada --------------------------------------------------------


@pytest.mark.parametrize(
    "fracciones",
    [(0.5, 0.3, 0.3), (0.5, 0.2, 0.2), (0.7, 0.3), (0.5, 0.25, 0.15, 0.1), (1.2, -0.1, -0.1)],
    ids=["suma_1.1", "suma_0.9", "solo_dos", "cuatro", "negativas"],
)
def test_fracciones_invalidas_lanzan_value_error(metadata_valida, fracciones):
    df = metadata_valida(n_lechugas=20, fotos_por_lechuga=1)

    with pytest.raises(ValueError, match="fracciones"):
        group_split(df, fracciones=fracciones)


def test_menos_de_tres_grupos_lanza_value_error(metadata_valida):
    df = metadata_valida(n_lechugas=2, fotos_por_lechuga=5)

    with pytest.raises(ValueError, match="grupos"):
        group_split(df)


def test_con_tres_grupos_ninguna_particion_queda_vacia(metadata_valida):
    df = metadata_valida(n_lechugas=3, fotos_por_lechuga=2)

    salida = group_split(df)

    assert set(salida["split"]) == PARTICIONES


def test_columna_de_grupo_inexistente_lanza_value_error(metadata_valida):
    df = metadata_valida(n_lechugas=10, fotos_por_lechuga=1)

    with pytest.raises(ValueError, match="no_existe"):
        group_split(df, group_col="no_existe")


# --- Caso 5: filas sin lechuga_id ------------------------------------------------------


def test_filas_sin_lechuga_id_avisan_y_se_marcan(metadata_valida):
    df = metadata_valida(n_lechugas=20, fotos_por_lechuga=1)
    df.loc[[0, 1, 2], "lechuga_id"] = ["", "   ", None]

    with pytest.warns(UserWarning, match="3 filas sin lechuga_id") as registro:
        salida = group_split(df)

    assert "fuga" in str(registro[0].message)
    assert list(salida.index[salida["grupo_inferido"]]) == [0, 1, 2]
    assert salida["grupo_inferido"].sum() == 3


def test_cada_fila_sin_id_forma_su_propio_grupo(metadata_valida):
    df = metadata_valida(n_lechugas=10, fotos_por_lechuga=1)
    df["lechuga_id"] = None  # las 10 filas sin id: deben ser 10 grupos distintos

    with pytest.warns(UserWarning, match="10 filas sin lechuga_id"):
        salida = group_split(df)

    # Si fueran un solo grupo, todas caerían en la misma partición.
    assert set(salida["split"]) == PARTICIONES
    assert salida["grupo_inferido"].all()


def test_sin_filas_sin_id_no_hay_aviso_y_nada_se_marca(metadata_valida):
    df = metadata_valida(n_lechugas=20, fotos_por_lechuga=2)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        salida = group_split(df)

    assert not salida["grupo_inferido"].any()


def test_un_id_real_y_una_fila_sin_id_no_se_mezclan(metadata_valida):
    # Una lechuga llamada "0" no debe confundirse con la clave interna de una fila sin id.
    df = metadata_valida(n_lechugas=10, fotos_por_lechuga=1)
    df.loc[0, "lechuga_id"] = "0"
    df.loc[1, "lechuga_id"] = None

    with pytest.warns(UserWarning):
        salida = group_split(df)

    assert salida["grupo_inferido"].tolist() == [False, True] + [False] * 8


# --- Caso 11: no modifica el original --------------------------------------------------


def test_el_dataframe_original_no_cambia(metadata_valida):
    df = metadata_valida(n_lechugas=20, fotos_por_lechuga=2)
    df.loc[0, "lechuga_id"] = None
    antes = df.copy(deep=True)

    with pytest.warns(UserWarning):
        salida = group_split(df)

    pd.testing.assert_frame_equal(df, antes)
    assert "split" not in df.columns and "grupo_inferido" not in df.columns
    assert salida is not df
    assert list(salida.index) == list(df.index)
    assert {"split", "grupo_inferido"} <= set(salida.columns)
