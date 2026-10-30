"""Pruebas de src/data/schema.py: esquema de metadatos y validación."""

import numpy as np
import pandas as pd
import pytest

from src.data.schema import (
    CONSERVACIONES,
    GLOBAL_LEVELS,
    GROUPS,
    METADATA_COLUMNS,
    NO_VISIBLE,
    VISTAS,
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
VISTAS_ESPERADAS = ("cenital", "lateral_a", "lateral_b", "base")
CONSERVACIONES_ESPERADAS = ("nevera", "ambiente")
# Columnas de contexto de captura: opcionales (pueden ir vacías) pero siempre en la cabecera.
COLUMNAS_CONTEXTO = (
    "variedad", "estado_seq", "horas_desde_corte", "conservacion", "temperatura_c",
    "intervencion", "dispositivo", "archivo_original",
)  # fmt: skip
COLUMNAS_TEXTO = ("variedad", "intervencion", "dispositivo", "archivo_original")
COLUMNAS_ESPERADAS = (
    "image_id", "fuente", "licencia", "etiqueta_original", "lechuga_id", "vista", "sesion",
    "corte", "recorte", "hojas_ext", "color", "daño_fisico", "daño_bio", "salida_global",
    *COLUMNAS_CONTEXTO,
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


def test_vistas_y_conservaciones_coinciden_con_el_brief():
    assert VISTAS == VISTAS_ESPERADAS
    assert CONSERVACIONES == CONSERVACIONES_ESPERADAS


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
    df.loc[0, "color"] = "VERDE"  # con errores, para que el validador trabaje
    df.loc[1, "estado_seq"] = -1
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

    assert errores == ["Falta 1 columna: licencia"]


def test_columnas_ausentes_se_avisan_en_un_solo_error_que_las_nombra(metadata_valida):
    df = metadata_valida().drop(columns=["daño_bio", "salida_global"])

    errores = validate_metadata(df)  # no lanza aunque falten columnas

    assert errores == ["Faltan 2 columnas: daño_bio, salida_global"]


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


# --- Columnas de contexto de captura: cabecera -----------------------------------------


@pytest.mark.parametrize("columna", COLUMNAS_CONTEXTO)
def test_falta_una_columna_de_contexto_y_el_error_la_nombra(metadata_valida, columna):
    df = metadata_valida().drop(columns=[columna])

    assert validate_metadata(df) == [f"Falta 1 columna: {columna}"]


def test_faltan_todas_las_columnas_de_contexto_en_un_solo_error_y_en_orden(metadata_valida):
    df = metadata_valida().drop(columns=list(COLUMNAS_CONTEXTO))

    errores = validate_metadata(df)

    assert errores == [f"Faltan 8 columnas: {', '.join(COLUMNAS_CONTEXTO)}"]


def test_con_columnas_ausentes_se_siguen_validando_los_valores(metadata_valida):
    df = metadata_valida().drop(columns=["temperatura_c"])
    df.loc[1, "vista"] = "superior"

    errores = validate_metadata(df)

    assert len(errores) == 2
    assert any(e.startswith("Falta") and "temperatura_c" in e for e in errores)
    assert any("img_0001" in e and "vista" in e and "superior" in e for e in errores)


# --- Columnas de contexto de captura: vacías y rellenas --------------------------------


@pytest.mark.parametrize("vacio", ["", "   ", None, float("nan")], ids=repr)
@pytest.mark.parametrize("columna", COLUMNAS_CONTEXTO + ("vista",))
def test_las_columnas_de_contexto_pueden_ir_vacias(metadata_valida, columna, vacio):
    df = metadata_valida()
    df.loc[0, columna] = vacio

    assert validate_metadata(df) == []


def test_todas_las_columnas_de_contexto_rellenas_con_valores_validos(metadata_valida):
    df = metadata_valida()
    fila_0 = {
        "variedad": "Little Gem",
        "estado_seq": 0,
        "horas_desde_corte": 0,
        "conservacion": "nevera",
        "temperatura_c": -2.5,
        "intervencion": "lavado",
        "dispositivo": "movil_a",
        "archivo_original": "IMG_0001.jpg",
    }
    fila_1 = {**fila_0, "estado_seq": 12, "horas_desde_corte": 36.5, "conservacion": "ambiente"}
    for columna in COLUMNAS_CONTEXTO:
        df.loc[0, columna] = fila_0[columna]
        df.loc[1, columna] = fila_1[columna]

    assert validate_metadata(df) == []


# --- vista y conservacion: lista cerrada o vacío ----------------------------------------


@pytest.mark.parametrize("vista", VISTAS_ESPERADAS)
def test_vista_valida(metadata_valida, vista):
    df = metadata_valida()
    df.loc[0, "vista"] = vista

    assert validate_metadata(df) == []


@pytest.mark.parametrize("vista", ["lateral", "Cenital", "superior", "lateral_c", " base"])
def test_vista_invalida_dice_columna_fila_y_valor(metadata_valida, vista):
    df = metadata_valida()
    df.loc[2, "vista"] = vista

    errores = validate_metadata(df)

    assert len(errores) == 1
    assert "img_0002" in errores[0] and "'vista'" in errores[0] and f"'{vista}'" in errores[0]


@pytest.mark.parametrize("conservacion", CONSERVACIONES_ESPERADAS)
def test_conservacion_valida(metadata_valida, conservacion):
    df = metadata_valida()
    df.loc[0, "conservacion"] = conservacion

    assert validate_metadata(df) == []


@pytest.mark.parametrize("conservacion", ["frigorifico", "Nevera", " nevera", "nevera "])
def test_conservacion_invalida_dice_columna_fila_y_valor(metadata_valida, conservacion):
    df = metadata_valida()
    df.loc[3, "conservacion"] = conservacion

    errores = validate_metadata(df)

    assert len(errores) == 1
    assert "img_0003" in errores[0] and "'conservacion'" in errores[0]
    assert f"'{conservacion}'" in errores[0]


# --- estado_seq, horas_desde_corte y temperatura_c: números -----------------------------


@pytest.mark.parametrize("valor", [0, 3, "3", 3.0, "3.0", np.int64(4), np.float64(5.0)], ids=repr)
def test_estado_seq_acepta_enteros_no_negativos(metadata_valida, valor):
    df = metadata_valida()
    df.loc[0, "estado_seq"] = valor

    assert validate_metadata(df) == []


@pytest.mark.parametrize(
    "valor", [-1, "-1", 2.5, "2.5", "abc", "nan", "inf", float("inf"), True, np.True_], ids=repr
)
def test_estado_seq_rechaza_negativos_decimales_y_no_numeros(metadata_valida, valor):
    df = metadata_valida()
    df.loc[1, "estado_seq"] = valor

    errores = validate_metadata(df)

    assert len(errores) == 1
    assert "img_0001" in errores[0] and "'estado_seq'" in errores[0]
    assert f"'{valor}'" in errores[0]


@pytest.mark.parametrize("valor", [0, 12.5, "12.5", "0", "7", np.float64(1.5)], ids=repr)
def test_horas_desde_corte_acepta_numeros_no_negativos(metadata_valida, valor):
    df = metadata_valida()
    df.loc[0, "horas_desde_corte"] = valor

    assert validate_metadata(df) == []


@pytest.mark.parametrize(
    "valor",
    [-0.5, "-1", "12,5", "abc", "nan", "inf", float("inf"), float("-inf"), "1_0", True],
    ids=repr,
)
def test_horas_desde_corte_rechaza_negativos_y_no_numeros(metadata_valida, valor):
    df = metadata_valida()
    df.loc[4, "horas_desde_corte"] = valor

    errores = validate_metadata(df)

    assert len(errores) == 1
    assert "img_0004" in errores[0] and "'horas_desde_corte'" in errores[0]
    assert f"'{valor}'" in errores[0]


@pytest.mark.parametrize("valor", [4, 0, -2.5, "21.3", "-18", "+3.5", ".5"], ids=repr)
def test_temperatura_acepta_cualquier_numero_incluidos_negativos(metadata_valida, valor):
    df = metadata_valida()
    df.loc[0, "temperatura_c"] = valor

    assert validate_metadata(df) == []


@pytest.mark.parametrize(
    "valor", ["fría", "4,5", "nan", "inf", float("inf"), "1_0", "12 grados", True], ids=repr
)
def test_temperatura_rechaza_no_numeros(metadata_valida, valor):
    df = metadata_valida()
    df.loc[5, "temperatura_c"] = valor

    errores = validate_metadata(df)

    assert len(errores) == 1
    assert "img_0005" in errores[0] and "'temperatura_c'" in errores[0]
    assert f"'{valor}'" in errores[0]


# --- Columnas de texto libre -----------------------------------------------------------


@pytest.mark.parametrize("valor", ["Lechuga baby ñ", "áéíóú", "IMG_001.jpg", 7, 3.5], ids=repr)
@pytest.mark.parametrize("columna", COLUMNAS_TEXTO)
def test_columnas_de_texto_libre_aceptan_cualquier_valor(metadata_valida, columna, valor):
    df = metadata_valida()
    df.loc[0, columna] = valor

    assert validate_metadata(df) == []


def test_varios_valores_invalidos_dan_un_error_por_problema(metadata_valida):
    df = metadata_valida()
    malos = [
        (0, "vista", "superior"),
        (1, "estado_seq", -1),
        (2, "horas_desde_corte", "12,5"),
        (3, "conservacion", "frigorifico"),
        (4, "temperatura_c", "fría"),
    ]
    for fila, columna, valor in malos:
        df.loc[fila, columna] = valor

    errores = validate_metadata(df)

    assert len(errores) == len(malos)
    for fila, columna, valor in malos:
        assert any(
            f"img_{fila:04d}" in e and f"'{columna}'" in e and f"'{valor}'" in e for e in errores
        ), (fila, columna, valor, errores)


def test_los_mensajes_se_pueden_imprimir_en_una_consola_de_windows(metadata_valida):
    # Si un mensaje llevara un carácter fuera de cp1252 (p. ej. "≥"), imprimirlo con la
    # salida redirigida en Windows lanzaría UnicodeEncodeError.
    df = metadata_valida().drop(columns=["temperatura_c"])
    df.loc[0, ["vista", "conservacion"]] = ["superior", "frigorifico"]
    df.loc[1, ["estado_seq", "horas_desde_corte"]] = [-1, -1]

    errores = validate_metadata(df)

    assert len(errores) == 5
    "\n".join(errores).encode("cp1252")  # no debe lanzar


def test_valor_invalido_sin_image_id_se_identifica_por_la_fila(metadata_valida):
    df = metadata_valida()
    df.loc[2, "image_id"] = ""
    df.loc[2, "estado_seq"] = -1

    errores = [e for e in validate_metadata(df) if "estado_seq" in e]

    assert len(errores) == 1 and "fila 2" in errores[0] and "'-1'" in errores[0]


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


def test_csv_con_contexto_relleno_ida_y_vuelta_en_utf8(metadata_valida, tmp_path):
    df = metadata_valida()
    df.loc[0, ["variedad", "intervencion", "archivo_original"]] = ["Ñora", "lavado", "foto_ñ.jpg"]
    df.loc[0, ["estado_seq", "horas_desde_corte", "temperatura_c"]] = ["2", "12.5", "-1.5"]
    df.loc[0, "conservacion"] = "nevera"
    ruta = tmp_path / "metadata.csv"
    df.to_csv(ruta, index=False, encoding="utf-8")

    leido = pd.read_csv(ruta, encoding="utf-8", dtype=str, keep_default_na=False)

    assert list(leido.columns) == list(METADATA_COLUMNS)
    assert leido.loc[0, "variedad"] == "Ñora"
    assert validate_metadata(leido) == []


def test_csv_leido_con_pandas_por_defecto_es_valido(metadata_valida, tmp_path):
    # pandas convierte una columna entera con huecos en float (2 -> 2.0, hueco -> NaN):
    # eso no debe dar error en estado_seq.
    df = metadata_valida()
    df.loc[[0, 1], "estado_seq"] = [0, 2]
    df.loc[[0, 1], "temperatura_c"] = ["4.5", "-2"]
    ruta = tmp_path / "metadata.csv"
    df.to_csv(ruta, index=False, encoding="utf-8")

    leido = pd.read_csv(ruta, encoding="utf-8")

    assert str(leido["estado_seq"].dtype) == "float64"
    assert leido["variedad"].isna().all()
    assert validate_metadata(leido) == []
