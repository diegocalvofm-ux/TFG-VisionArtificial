"""Pruebas de src/data/schema.py: esquema de metadatos y validación."""

import pandas as pd
import pytest

from src.data.schema import (
    GLOBAL_LEVELS,
    GROUPS,
    METADATA_COLUMNS,
    NO_VISIBLE,
    is_blank,
    validate_metadata,
)

# Esquema del brief escrito a mano: si alguien lo edita sin querer, esta prueba avisa.
GROUPS_ESPERADOS = {
    "corte": ("CUT_OK", "STEM_TOO_LONG", "UNCLEAN_CUT", "ROOT_REMAINS"),
    "recorte": ("TRIM_OK", "UNDER_TRIMMED", "OVER_TRIMMED"),
    "hojas_ext": ("OUTER_LEAVES_OK", "OUTER_DAMAGE_LIGHT", "OUTER_DAMAGE_MAJOR"),
    "color": ("COLOR_FRESH_OK", "YELLOW_BROWN_LIGHT", "YELLOW_BROWN_MAJOR", "WILTING"),
    "daño_fisico": ("PHYSICAL_OK", "TORN_LEAF", "BRUISE_CRUSH"),
    "daño_bio": ("BIO_OK", "ROT_DECAY", "VISIBLE_PEST", "PEST_DAMAGE", "DISEASE_VISIBLE"),
}
COLUMNAS_ESPERADAS = (
    "image_id", "fuente", "licencia", "etiqueta_original", "lechuga_id", "vista", "sesion",
    "corte", "recorte", "hojas_ext", "color", "daño_fisico", "daño_bio", "salida_global",
)  # fmt: skip


# --- El esquema en sí -------------------------------------------------------------------


def test_groups_coincide_con_el_brief():
    assert GROUPS == GROUPS_ESPERADOS
    assert all(isinstance(valores, tuple) for valores in GROUPS.values())


def test_no_visible_es_constante_aparte_y_no_esta_en_groups():
    assert NO_VISIBLE == "NO_VISIBLE"
    assert all(NO_VISIBLE not in valores for valores in GROUPS.values())


def test_columnas_coinciden_con_el_brief():
    assert tuple(METADATA_COLUMNS) == COLUMNAS_ESPERADAS


def test_niveles_globales():
    assert tuple(GLOBAL_LEVELS) == ("ACCEPT", "REVIEW", "NON-CONFORMING")


@pytest.mark.parametrize("valor", [None, float("nan"), pd.NA, "", "   "])
def test_is_blank_detecta_vacios(valor):
    assert is_blank(valor)


@pytest.mark.parametrize("valor", ["a", "NO_VISIBLE", 0, 1.5])
def test_is_blank_no_confunde_valores_reales(valor):
    assert not is_blank(valor)


# --- validate_metadata: caso feliz -----------------------------------------------------


def test_metadata_valida_no_da_errores(metadata_valida):
    assert validate_metadata(metadata_valida()) == []


def test_validate_metadata_no_modifica_el_dataframe(metadata_valida):
    df = metadata_valida()
    df.loc[0, "color"] = "VERDE"  # con un error, para que el validador trabaje
    antes = df.copy(deep=True)

    validate_metadata(df)

    pd.testing.assert_frame_equal(df, antes)


def test_dataframe_sin_filas_solo_comprueba_columnas(metadata_valida):
    assert validate_metadata(metadata_valida().iloc[0:0]) == []


# --- Caso 6: valor inválido en un grupo ------------------------------------------------


def test_valor_invalido_en_un_grupo_se_detecta(metadata_valida):
    df = metadata_valida()
    df.loc[2, "color"] = "VERDE"

    errores = validate_metadata(df)

    assert len(errores) == 1
    assert "img_0002" in errores[0] and "color" in errores[0] and "VERDE" in errores[0]


def test_valor_de_otro_grupo_es_invalido(metadata_valida):
    df = metadata_valida()
    df.loc[0, "corte"] = "BIO_OK"  # existe, pero en otro grupo

    errores = validate_metadata(df)

    assert len(errores) == 1 and "BIO_OK" in errores[0]


def test_la_comparacion_distingue_mayusculas(metadata_valida):
    df = metadata_valida()
    df.loc[0, "corte"] = "cut_ok"

    assert len(validate_metadata(df)) == 1


# --- Caso 7: NO_VISIBLE vale en cualquier grupo ----------------------------------------


@pytest.mark.parametrize("grupo", list(GROUPS_ESPERADOS))
def test_no_visible_se_acepta_en_los_seis_grupos(metadata_valida, grupo):
    df = metadata_valida()
    df.loc[0, grupo] = "NO_VISIBLE"

    assert validate_metadata(df) == []


# --- Caso 8: fuente, licencia, image_id duplicado y columna ausente --------------------


@pytest.mark.parametrize("columna", ["fuente", "licencia"])
@pytest.mark.parametrize("vacio", ["", "   ", None], ids=["cadena_vacia", "espacios", "nulo"])
def test_fuente_o_licencia_vacia_se_detecta(metadata_valida, columna, vacio):
    df = metadata_valida()
    df.loc[1, columna] = vacio

    errores = validate_metadata(df)

    assert len(errores) == 1
    assert "img_0001" in errores[0] and columna in errores[0]


def test_image_id_duplicado_se_detecta(metadata_valida):
    df = metadata_valida()
    df.loc[1, "image_id"] = df.loc[0, "image_id"]

    errores = validate_metadata(df)

    assert len(errores) == 1
    assert "img_0000" in errores[0] and "duplicad" in errores[0]


def test_image_id_vacio_se_detecta_y_no_cuenta_como_duplicado(metadata_valida):
    df = metadata_valida()
    df.loc[0, "image_id"] = ""
    df.loc[1, "image_id"] = None

    errores = validate_metadata(df)

    assert len(errores) == 2
    assert all("image_id" in e and "fila" in e for e in errores)
    assert not any("duplicad" in e for e in errores)


def test_columna_ausente_se_detecta(metadata_valida):
    df = metadata_valida().drop(columns=["licencia"])

    errores = validate_metadata(df)

    assert len(errores) == 1 and "licencia" in errores[0]


def test_columna_de_grupo_ausente_se_detecta_y_no_lanza(metadata_valida):
    df = metadata_valida().drop(columns=["daño_bio", "salida_global"])

    errores = validate_metadata(df)

    assert len(errores) == 2
    assert any("daño_bio" in e for e in errores)
    assert any("salida_global" in e for e in errores)


# --- Caso 9: celda vacía en un grupo ---------------------------------------------------


@pytest.mark.parametrize("vacio", ["", "   ", None], ids=["cadena_vacia", "espacios", "nulo"])
def test_celda_vacia_en_un_grupo_se_detecta(metadata_valida, vacio):
    df = metadata_valida()
    df.loc[1, "corte"] = vacio

    errores = validate_metadata(df)

    assert len(errores) == 1
    assert "img_0001" in errores[0] and "corte" in errores[0] and NO_VISIBLE in errores[0]


# --- Caso 10: salida_global ------------------------------------------------------------


@pytest.mark.parametrize("vacio", ["", None], ids=["cadena_vacia", "nulo"])
def test_salida_global_vacia_es_valida(metadata_valida, vacio):
    df = metadata_valida()
    df.loc[0, "salida_global"] = vacio

    assert validate_metadata(df) == []


@pytest.mark.parametrize("nivel", GLOBAL_LEVELS)
def test_salida_global_con_nivel_valido(metadata_valida, nivel):
    df = metadata_valida()
    df.loc[0, "salida_global"] = nivel

    assert validate_metadata(df) == []


def test_salida_global_fuera_de_la_lista_es_invalida(metadata_valida):
    df = metadata_valida()
    df.loc[0, "salida_global"] = "QUIZAS"

    errores = validate_metadata(df)

    assert len(errores) == 1
    assert "img_0000" in errores[0] and "salida_global" in errores[0] and "QUIZAS" in errores[0]


# --- Varios problemas a la vez y CSV ---------------------------------------------------


def test_devuelve_un_error_por_problema(metadata_valida):
    df = metadata_valida()
    df.loc[0, "color"] = "VERDE"
    df.loc[3, "licencia"] = ""
    df.loc[4, "salida_global"] = "QUIZAS"

    assert len(validate_metadata(df)) == 3


def test_csv_con_enes_ida_y_vuelta_en_utf8(metadata_valida, tmp_path):
    ruta = tmp_path / "metadata.csv"
    metadata_valida().to_csv(ruta, index=False, encoding="utf-8")

    assert "daño_fisico".encode("utf-8") in ruta.read_bytes()
    leido = pd.read_csv(ruta, encoding="utf-8", dtype=str, keep_default_na=False)
    assert "daño_fisico" in leido.columns and "daño_bio" in leido.columns
    assert validate_metadata(leido) == []
