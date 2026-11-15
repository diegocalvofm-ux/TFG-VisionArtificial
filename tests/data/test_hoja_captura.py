"""Pruebas de src/data/hoja_captura.py: normalización de textos, recuentos y lectura del xlsx."""

from pathlib import Path

import pytest

from src.config import load_config
from src.data.hoja_captura import (
    es_no_hecha,
    leer_hoja,
    leer_recuento,
    leer_se_ve,
    normalizar,
    traducir_estado,
)
from src.data.propio import validar_reglas

RAIZ_REPO = Path(__file__).resolve().parents[2]
REGLAS = validar_reglas(load_config(RAIZ_REPO / "configs" / "datasets" / "propio_piloto.yaml"))


# --- Normalización y estados ------------------------------------------------------------


def test_normalizar_quita_mayusculas_tildes_signos_espacios_y_plurales():
    assert normalizar("  Pocas HOJAS   rasgadas ") == "poca hoja rasgada"
    assert normalizar("Sin raíz, con tallo.") == normalizar("sin raiz con tallos")
    assert normalizar("¿Se ve el daño?") == "se ve el dano"


def test_normalizar_no_quita_la_s_de_palabras_cortas():
    assert normalizar("las tres") == "las tre"  # "las" tiene 3 letras: se queda igual


@pytest.mark.parametrize(
    "texto, codigo",
    [
        ("Tal como viene", "0"),
        ("Sin raiz con tallos", "1"),
        ("Sín raíz con tallo", "1"),
        ("corte MALO", "2"),
        ("Corte bueno y hojas malas quitadas", "4"),
        ("Pocas hojas rasgada", "5"),
        ("POCAS  HOJAS RASGADAS ", "5"),
        ("pocas hojas rasgadas", "5"),
        ("Golpe", "6"),
        ("Muchas hojas dañadas", "7"),
        ("Pelada", "8"),
        ("24 h", "24h"),
        ("24H", "24h"),
        ("48 horas", "48h"),
        ("5", "5"),  # el código tal cual también vale
        (5, "5"),  # Excel puede guardarlo como número
        (5.0, "5"),
    ],
)
def test_traducir_estado_tolera_variaciones(texto, codigo):
    assert traducir_estado(texto, REGLAS.alias) == codigo


@pytest.mark.parametrize("texto", ["Golpe fuerte", "corte", "9", 9, 5.5, "", None])
def test_traducir_estado_desconocido_devuelve_none(texto):
    assert traducir_estado(texto, REGLAS.alias) is None


# --- Recuentos ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "valor, esperado",
    [
        (0, 0),
        (3, 3),
        (3.0, 3),  # Excel guarda los enteros como decimales a veces
        ("3", 3),
        (" 3 ", 3),
        ("~10", 10),
        ("~ 10", 10),
        ("1-2", 2),  # rango: se toma el máximo
        ("1 - 2", 2),
        ("2-1", 2),
        ("Todas", 5),
        ("todas", 5),
        ("TODAS", 5),
    ],
)
def test_leer_recuento_formas_admitidas(valor, esperado):
    assert leer_recuento(valor, REGLAS.palabras) == esperado


@pytest.mark.parametrize("vacio", [None, "", "   "])
def test_recuento_vacio_es_desconocido(vacio):
    assert leer_recuento(vacio, REGLAS.palabras) is None


@pytest.mark.parametrize("valor", ["abc", "muchas", -1, "-1", 2.5, "2.5", "~", "1-", True])
def test_recuento_no_reconocido_da_error(valor):
    with pytest.raises(ValueError, match="recuento"):
        leer_recuento(valor, REGLAS.palabras)


# --- ¿Se ve el daño? y filas con guiones -------------------------------------------------


@pytest.mark.parametrize(
    "valor, esperado",
    [("No", False), ("no", False), (" NO ", False), ("Sí", True), ("si", True), (None, None),
     ("", None)],
)  # fmt: skip
def test_leer_se_ve(valor, esperado):
    assert leer_se_ve(valor) is esperado


@pytest.mark.parametrize("valor", ["Poco", "a veces", 0])
def test_se_ve_con_otro_texto_da_error(valor):
    with pytest.raises(ValueError, match="no, sí o vacío"):
        leer_se_ve(valor)


@pytest.mark.parametrize(
    "celdas, esperado",
    [
        (["LB-001", "Batavia", "Corte malo", "----", "----", None], True),
        (["LB-001", "--"], True),
        (["LB-001", " —— "], True),  # guiones largos (autocorrección)
        (["LB-001", "-", 3], False),  # un solo guion no basta
        (["LB-001", "Corte malo", 0, None], False),
        ([], False),
    ],
)
def test_es_no_hecha(celdas, esperado):
    assert es_no_hecha(celdas) is esperado


# --- Lectura del xlsx ---------------------------------------------------------------------


def test_leer_hoja_devuelve_las_filas_con_su_numero(hoja_xlsx, fila_hoja):
    ruta = hoja_xlsx(
        [
            fila_hoja("LI-001", "Tal como viene", danadas=1, amarillas="~3", notas="ok"),
            fila_hoja("LI-001", "24 h", se_ve="No"),
        ]
    )

    filas = leer_hoja(ruta, "Captura", REGLAS.columnas)

    assert [f["fila"] for f in filas] == [2, 3]  # la fila 1 es la cabecera
    assert filas[0]["lechuga"] == "LI-001"
    assert filas[0]["variedad"] == "Iceberg"
    assert filas[0]["estado"] == "Tal como viene"
    assert filas[0]["hojas_danadas"] == 1
    assert filas[0]["hojas_amarillas"] == "~3"
    assert filas[0]["notas"] == "ok"
    assert filas[1]["se_ve_dano"] == "No"
    assert filas[0]["no_hecha"] is False


def test_cabecera_debajo_de_un_titulo(hoja_xlsx, fila_hoja):
    ruta = hoja_xlsx([fila_hoja("LI-001", "0")], titulo="Hoja de captura piloto")

    filas = leer_hoja(ruta, "Captura", REGLAS.columnas)

    assert [f["fila"] for f in filas] == [3]
    assert filas[0]["lechuga"] == "LI-001"


def test_cabeceras_se_comparan_sin_mayusculas_ni_tildes(hoja_xlsx, fila_hoja):
    cabeceras = (
        "LECHUGA", "variedad", "Que le has hecho (Estado)", "Fecha", "Hora",
        "hojas de fuera danadas (nº total)", "Hojas amarillas o marrones (nº)",
        "Se ve el dano en las fotos", "  notas ",
    )  # fmt: skip
    ruta = hoja_xlsx([fila_hoja("LB-001", "Golpe")], cabeceras=cabeceras)

    filas = leer_hoja(ruta, "Captura", REGLAS.columnas)

    assert filas[0]["estado"] == "Golpe"


def test_falta_la_hoja_da_error_que_la_nombra(hoja_xlsx, fila_hoja):
    ruta = hoja_xlsx([fila_hoja("LI-001", "0")], hoja="Hoja1")
    with pytest.raises(ValueError, match="'Captura'.*Hoja1"):
        leer_hoja(ruta, "Captura", REGLAS.columnas)


def test_falta_una_columna_da_error_que_la_nombra(hoja_xlsx, fila_hoja):
    sin_notas = (
        "Lechuga", "Variedad", "Qué le has hecho (estado)", "Fecha", "Hora",
        "Hojas de fuera dañadas (nº total)", "Hojas amarillas o marrones (nº)",
        "¿Se ve el daño en las fotos?",
    )  # fmt: skip
    ruta = hoja_xlsx([fila_hoja("LI-001", "0")[:-1]], cabeceras=sin_notas)
    with pytest.raises(ValueError, match="Notas"):
        leer_hoja(ruta, "Captura", REGLAS.columnas)


def test_celdas_combinadas_toman_el_valor_de_la_combinacion(hoja_xlsx, fila_hoja):
    ruta = hoja_xlsx(
        [fila_hoja("LB-001", "0"), fila_hoja("LB-001", "3"), fila_hoja("LB-001", "4")],
        combinar=("A2:A4", "B2:B4"),
    )

    filas = leer_hoja(ruta, "Captura", REGLAS.columnas)

    assert [f["lechuga"] for f in filas] == ["LB-001"] * 3
    assert [f["variedad"] for f in filas] == ["Batavia"] * 3


def test_filas_vacias_se_saltan(hoja_xlsx, fila_hoja):
    ruta = hoja_xlsx([fila_hoja("LI-001", "0"), [None] * 9, fila_hoja("LI-001", "24h")])

    filas = leer_hoja(ruta, "Captura", REGLAS.columnas)

    assert [f["fila"] for f in filas] == [2, 4]


def test_fila_con_guiones_en_columnas_no_usadas_se_marca_no_hecha(hoja_xlsx, fila_hoja):
    # Los guiones están en Fecha y Hora, que la ingesta no lee: cuentan igual.
    ruta = hoja_xlsx([fila_hoja("LB-001", "Corte malo", fecha="----", hora="----")])

    filas = leer_hoja(ruta, "Captura", REGLAS.columnas)

    assert filas[0]["no_hecha"] is True
