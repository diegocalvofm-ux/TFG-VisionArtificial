"""Pruebas de src/data/propio.py: ingesta de las fotos propias (datos sintéticos, nunca reales)."""

import copy
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.data.propio import (
    FilaHoja,
    etiquetas_de_estado,
    etiquetas_de_foto,
    ingerir,
    leer_fotos,
    validar_reglas,
)
from src.data.schema import GROUPS, METADATA_COLUMNS, NO_VISIBLE, validate_metadata
from src.evaluation.aggregate import aggregate

RAIZ_REPO = Path(__file__).resolve().parents[2]
TABLA = load_config(RAIZ_REPO / "configs" / "salida_global.yaml")
CONFIG_PILOTO = load_config(RAIZ_REPO / "configs" / "datasets" / "propio_piloto.yaml")
REGLAS = validar_reglas(CONFIG_PILOTO)
VISTAS = ("C", "F", "G", "B")

# Matriz de visibilidad aprobada (grupo -> vistas en que se ve), escrita a mano a propósito.
MATRIZ = {
    "corte": {"B"},
    "recorte": {"C", "F", "G", "B"},
    "hojas_ext": {"C", "F", "G", "B"},
    "color": {"C", "F", "G", "B"},
    "daño_fisico": {"C", "F", "G"},
    "daño_bio": {"C", "F", "G", "B"},
}


def _cuatro(lechuga: str, estado: str, vistas=VISTAS) -> list[str]:
    """Nombres de las fotos de una lechuga en un estado (por defecto, las cuatro vistas)."""
    return [f"{lechuga}_{estado}_{v}.jpg" for v in vistas]


def _ingerir(ruta_yaml: Path, raiz: Path, comprobar: bool = False) -> dict:
    return ingerir(
        ruta_yaml, raiz=raiz, metadata=raiz / "data" / "metadata.csv", tabla=TABLA,
        comprobar=comprobar,
    )  # fmt: skip


def _metadata(raiz: Path) -> pd.DataFrame:
    return pd.read_csv(
        raiz / "data" / "metadata.csv", dtype=str, keep_default_na=False, encoding="utf-8"
    )


def _fila_publica(image_id: str, **valores) -> dict:
    """Fila válida de metadata.csv de otra fuente (grupos en NO_VISIBLE)."""
    fila = dict.fromkeys(METADATA_COLUMNS, "")
    fila.update(dict.fromkeys(GROUPS, NO_VISIBLE))
    fila.update(image_id=image_id, fuente="giovi", licencia="CC BY 4.0", salida_global="REVIEW")
    fila.update(valores)
    return fila


def _escribir_metadata(raiz: Path, filas: list[dict]) -> Path:
    ruta = raiz / "data" / "metadata.csv"
    ruta.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(filas, columns=list(METADATA_COLUMNS)).to_csv(
        ruta, index=False, encoding="utf-8", lineterminator="\n"
    )
    return ruta


def _fila(lechuga="LI-001", estado="0", danadas=0, amarillas=0, se_ve=None) -> FilaHoja:
    """Fila de la hoja ya traducida, para probar las reglas sin pasar por el xlsx."""
    return FilaHoja(
        fila=2,
        lechuga=lechuga,
        variedad=REGLAS.variedades[lechuga[:2]],
        estado=estado,
        estado_texto=estado,
        recuentos={"hojas_danadas": danadas, "hojas_amarillas": amarillas},
        se_ve=se_ve,
        notas="",
    )


def _base(**kwargs) -> dict:
    """Etiquetas de la lechuga en ese estado, antes de aplicar la visibilidad de cada vista."""
    return etiquetas_de_estado(_fila(**kwargs), REGLAS)[0]


# --- Configuración ------------------------------------------------------------------------


def test_el_yaml_del_piloto_cumple_las_reglas_y_la_matriz_aprobada():
    assert {g: set(v) for g, v in REGLAS.visibilidad.items()} == MATRIZ
    assert REGLAS.fuente == "propio" and REGLAS.sesion == "piloto"
    assert REGLAS.licencia == "propia (titularidad por confirmar)"
    assert set(REGLAS.estados) == {*map(str, range(9)), "24h", "48h"}


def _quitar(ruta: tuple, clave):
    def cambio(config):
        destino = config
        for paso in ruta:
            destino = destino[paso]
        del destino[clave]

    return cambio


def _poner(ruta: tuple, clave, valor):
    def cambio(config):
        destino = config
        for paso in ruta:
            destino = destino[paso]
        destino[clave] = valor

    return cambio


@pytest.mark.parametrize(
    "cambio, mensaje",
    [
        (_quitar(("por_estado",), "daño_bio"), "daño_bio"),
        (_quitar(("por_estado", "corte"), "3"), "corte.*'3'"),
        (_poner(("por_estado", "corte"), "3", "CUT_GOOD"), "CUT_GOOD"),
        (_poner(("por_estado", "corte"), "3", "TRIM_OK"), "TRIM_OK"),  # etiqueta de otro grupo
        (_poner(("por_estado", "daño_bio"), "0", "estado_0"), "estado_0"),
        (_poner(("visibilidad",), "corte", ["B", "X"]), "'X'"),
        (_quitar(("visibilidad",), "color"), "visibilidad.*color"),
        (_poner(("vistas",), "C", "trasera"), "trasera"),
        (_poner(("vistas",), "G", "lateral_a"), "lateral_a"),  # dos letras, misma vista
        (_poner(("conservacion",), "LI-001", "congelador"), "congelador"),
        (_poner(("estados",), "6", ["golpe", "Pelada"]), "pelada"),  # alias repetido
        (_poner(("recuentos",), "leve_desde", 5), "leve_desde"),
        (_poner(("recuentos", "grupos", "color"), "columna", "otra"), "otra"),
        (_poner(("recuentos", "grupos", "color"), "etiquetas", ["COLOR_FRESH_OK"]), "3 etiquetas"),
        (_poner(("por_estado",), "color", {}), "color"),  # grupo en por_estado y en recuentos
        (_poner((), "visibilidd", {}), "visibilidd"),  # clave desconocida (errata)
        (_poner(("estados",), "X", ["raro"]), "'X'"),  # ni número ni de la serie
        (_poner(("no_se_ve",), "sin_defecto", "BIO_OK"), "BIO_OK"),
        (_quitar(("columnas",), "notas"), "notas"),
    ],
)
def test_config_no_valida_da_error_que_lo_explica(cambio, mensaje):
    config = copy.deepcopy(CONFIG_PILOTO)
    cambio(config)
    with pytest.raises(ValueError, match=mensaje):
        validar_reglas(config)


# --- Nombres de las fotos -----------------------------------------------------------------


def test_nombres_validos(tmp_path, jpg_sintetico):
    for nombre in ("LI-001_0_C.jpg", "LB-012_24h_B.JPG", "LB-001_8_G.jpeg"):
        jpg_sintetico(tmp_path / nombre)

    fotos, errores = leer_fotos(tmp_path, REGLAS)

    assert errores == []
    resumen = [(f.image_id, f.lechuga, f.estado, f.letra, f.nombre) for f in fotos]
    assert resumen == [
        ("LB-001_8_G", "LB-001", "8", "G", "LB-001_8_G.jpeg"),
        ("LB-012_24h_B", "LB-012", "24h", "B", "LB-012_24h_B.JPG"),
        ("LI-001_0_C", "LI-001", "0", "C", "LI-001_0_C.jpg"),
    ]


def test_nombres_no_validos_se_nombran_todos_en_un_solo_error(tmp_path, jpg_sintetico):
    malos = [
        "LX-001_0_C.jpg", "LI-01_0_C.jpg", "LI-001_9_C.jpg", "LI-001_12h_C.jpg",
        "LI-001_0_D.jpg", "LI-001_0.jpg", "LI-001 _0_C.jpg", "li-002_0_C.jpg", "notas.txt",
    ]  # fmt: skip
    # (li-002 y no li-001: en Windows los nombres no distinguen mayúsculas y chocarían.)
    for nombre in malos:
        jpg_sintetico(tmp_path / nombre)
    (tmp_path / "subcarpeta").mkdir()
    jpg_sintetico(tmp_path / "LI-001_0_C.jpg")

    fotos, errores = leer_fotos(tmp_path, REGLAS)

    assert [f.nombre for f in fotos] == ["LI-001_0_C.jpg"]  # la buena se lee igual
    assert len(errores) == 1
    for nombre in [*malos, "subcarpeta"]:
        assert nombre in errores[0]


def test_mismo_image_id_con_dos_extensiones_da_error(tmp_path, jpg_sintetico):
    jpg_sintetico(tmp_path / "LI-001_0_C.jpg")
    jpg_sintetico(tmp_path / "LI-001_0_C.jpeg")

    _, errores = leer_fotos(tmp_path, REGLAS)

    assert len(errores) == 1
    assert "LI-001_0_C" in errores[0] and "repetid" in errores[0]


# --- Reglas por estado --------------------------------------------------------------------


@pytest.mark.parametrize(
    "estado, corte",
    [("1", "STEM_TOO_LONG"), ("2", "UNCLEAN_CUT"),
     *[(str(e), "CUT_OK") for e in range(3, 9)]],
)  # fmt: skip
def test_corte_por_estado(estado, corte):
    assert _base(lechuga="LB-001", estado=estado)["corte"] == corte


def test_estado_0_depende_de_la_variedad():
    iceberg, batavia = _base(lechuga="LI-001"), _base(lechuga="LB-001")
    assert (iceberg["corte"], iceberg["recorte"]) == ("CUT_OK", "TRIM_OK")
    assert (batavia["corte"], batavia["recorte"]) == ("ROOT_REMAINS", "UNDER_TRIMMED")


@pytest.mark.parametrize("estado", ["24h", "48h"])
@pytest.mark.parametrize(
    "lechuga, corte, recorte",
    [("LI-001", "CUT_OK", "TRIM_OK"), ("LB-001", "ROOT_REMAINS", "UNDER_TRIMMED")],
)
def test_serie_hereda_corte_y_recorte_del_estado_0(estado, lechuga, corte, recorte):
    etiquetas = _base(lechuga=lechuga, estado=estado)
    assert (etiquetas["corte"], etiquetas["recorte"]) == (corte, recorte)


@pytest.mark.parametrize(
    "lechuga, estado, recorte",
    [
        ("LB-001", "1", "UNDER_TRIMMED"),  # 1 y 2: como el estado 0 de la variedad
        ("LB-001", "2", "UNDER_TRIMMED"),
        ("LI-001", "1", "TRIM_OK"),
        ("LI-001", "2", "TRIM_OK"),
        ("LB-001", "3", "UNDER_TRIMMED"),
        *[("LB-001", str(e), "TRIM_OK") for e in range(4, 8)],
        ("LB-001", "8", "OVER_TRIMMED"),
    ],
)
def test_recorte_por_estado(lechuga, estado, recorte):
    assert _base(lechuga=lechuga, estado=estado)["recorte"] == recorte


@pytest.mark.parametrize(
    "estado, dano",
    [*[(str(e), "PHYSICAL_OK") for e in range(5)], ("5", "TORN_LEAF"),
     ("6", "BRUISE_CRUSH"), ("7", "BRUISE_CRUSH"), ("8", "BRUISE_CRUSH"),
     ("24h", "PHYSICAL_OK"), ("48h", "PHYSICAL_OK")],
)  # fmt: skip
def test_dano_fisico_por_estado(estado, dano):
    assert _base(lechuga="LB-001", estado=estado)["daño_fisico"] == dano


@pytest.mark.parametrize("estado", [*map(str, range(9)), "24h", "48h"])
def test_dano_bio_siempre_bio_ok(estado):
    assert _base(lechuga="LB-001", estado=estado)["daño_bio"] == "BIO_OK"


@pytest.mark.parametrize(
    "n, hojas_ext, color",
    [
        (0, "OUTER_LEAVES_OK", "COLOR_FRESH_OK"),
        (2, "OUTER_LEAVES_OK", "COLOR_FRESH_OK"),
        (3, "OUTER_DAMAGE_LIGHT", "YELLOW_BROWN_LIGHT"),
        (4, "OUTER_DAMAGE_LIGHT", "YELLOW_BROWN_LIGHT"),
        (5, "OUTER_DAMAGE_MAJOR", "YELLOW_BROWN_MAJOR"),
        (12, "OUTER_DAMAGE_MAJOR", "YELLOW_BROWN_MAJOR"),
    ],
)
def test_umbrales_de_los_recuentos(n, hojas_ext, color):
    etiquetas = _base(danadas=n, amarillas=n)
    assert (etiquetas["hojas_ext"], etiquetas["color"]) == (hojas_ext, color)


def test_recuento_desconocido_deja_el_grupo_no_visible():
    etiquetas, desconocidos = etiquetas_de_estado(_fila(danadas=None, amarillas=1), REGLAS)
    assert etiquetas["hojas_ext"] == NO_VISIBLE
    assert etiquetas["color"] == "COLOR_FRESH_OK"
    assert desconocidos == ["hojas_ext"]


# --- Visibilidad y "No" -------------------------------------------------------------------


@pytest.mark.parametrize("letra", VISTAS)
@pytest.mark.parametrize("grupo", list(GROUPS))
def test_matriz_de_visibilidad(grupo, letra):
    base = _base(lechuga="LB-001", estado="5")  # ninguna etiqueta es NO_VISIBLE de partida
    etiquetas = etiquetas_de_foto(base, letra, None, REGLAS)
    esperado = base[grupo] if letra in MATRIZ[grupo] else NO_VISIBLE
    assert etiquetas[grupo] == esperado


def test_no_con_defecto_pone_dano_fisico_no_visible_en_todas_las_vistas():
    base = _base(lechuga="LB-001", estado="5")  # TORN_LEAF
    for letra in VISTAS:
        etiquetas = etiquetas_de_foto(base, letra, False, REGLAS)
        assert etiquetas["daño_fisico"] == NO_VISIBLE
        assert etiquetas["daño_bio"] == "BIO_OK"  # el resto no cambia


def test_no_sin_defecto_no_cambia_dano_fisico():
    base = _base(lechuga="LI-001", estado="0")  # PHYSICAL_OK
    assert etiquetas_de_foto(base, "C", False, REGLAS)["daño_fisico"] == "PHYSICAL_OK"


@pytest.mark.parametrize("se_ve", [True, None])
def test_si_o_vacio_no_cambian_nada(se_ve):
    base = _base(lechuga="LB-001", estado="6")
    assert etiquetas_de_foto(base, "C", se_ve, REGLAS)["daño_fisico"] == "BRUISE_CRUSH"


# --- Ingesta completa: avisos y errores ---------------------------------------------------


def test_toma_con_3_fotos_avisa_y_se_ingiere(ingesta_sintetica, fila_hoja, tmp_path):
    ruta = ingesta_sintetica(_cuatro("LB-001", "4", "FGB"), [fila_hoja("LB-001", "4")])

    informe = _ingerir(ruta, tmp_path)

    assert informe["vistas_incompletas"] == [{"lechuga": "LB-001", "estado": "4", "faltan": ["C"]}]
    assert len(_metadata(tmp_path)) == 3
    assert sorted(p.name for p in (tmp_path / "processed" / "propio").glob("*.jpg")) == [
        "LB-001_4_B.jpg", "LB-001_4_F.jpg", "LB-001_4_G.jpg",
    ]  # fmt: skip


def test_foto_sin_fila_da_error_que_la_nombra_y_no_escribe_nada(
    ingesta_sintetica, fila_hoja, tmp_path
):
    ruta = ingesta_sintetica(
        [*_cuatro("LI-001", "0"), "LI-002_0_C.jpg"], [fila_hoja("LI-001", "0")]
    )
    csv = _escribir_metadata(tmp_path, [_fila_publica("giovi_0001")])
    antes = csv.read_bytes()

    with pytest.raises(ValueError, match="LI-002_0_C.jpg"):
        _ingerir(ruta, tmp_path)

    assert csv.read_bytes() == antes
    assert not (tmp_path / "processed" / "propio").exists()


def test_foto_de_una_fila_con_guiones_es_foto_sin_fila(ingesta_sintetica, fila_hoja, tmp_path):
    ruta = ingesta_sintetica(
        _cuatro("LB-001", "2"),
        [fila_hoja("LB-001", "Corte malo", fecha="----", hora="----", danadas="----")],
    )
    with pytest.raises(ValueError, match=r"LB-001_2_B\.jpg.*guiones"):
        _ingerir(ruta, tmp_path)


def test_fila_sin_fotos_avisa_y_las_de_guiones_se_ignoran(ingesta_sintetica, fila_hoja, tmp_path):
    ruta = ingesta_sintetica(
        _cuatro("LI-001", "0"),
        [
            fila_hoja("LI-001", "0"),
            fila_hoja("LI-001", "24h"),
            fila_hoja("LI-001", "Corte malo", fecha="----"),
        ],
    )

    informe = _ingerir(ruta, tmp_path)

    assert informe["filas_sin_fotos"] == [{"fila": 3, "lechuga": "LI-001", "estado": "24h"}]
    assert informe["filas_ignoradas"] == [{"fila": 4, "lechuga": "LI-001", "estado": "2"}]


def test_recuento_desconocido_avisa_y_deja_no_visible(ingesta_sintetica, fila_hoja, tmp_path):
    ruta = ingesta_sintetica(_cuatro("LI-001", "0"), [fila_hoja("LI-001", "0", amarillas=None)])

    informe = _ingerir(ruta, tmp_path)

    assert informe["recuentos_desconocidos"] == [
        {"lechuga": "LI-001", "estado": "0", "grupo": "color",
         "columna": "Hojas amarillas o marrones (nº)"},
    ]  # fmt: skip
    assert set(_metadata(tmp_path)["color"]) == {NO_VISIBLE}


def test_no_en_la_hoja(ingesta_sintetica, fila_hoja, tmp_path):
    ruta = ingesta_sintetica(
        [*_cuatro("LI-001", "0"), *_cuatro("LB-001", "5")],
        [fila_hoja("LI-001", "0", se_ve="No"), fila_hoja("LB-001", "5", se_ve="No")],
    )

    informe = _ingerir(ruta, tmp_path)

    df = _metadata(tmp_path).set_index("image_id")
    assert set(df.loc[[f"LB-001_5_{v}" for v in VISTAS], "daño_fisico"]) == {NO_VISIBLE}
    assert df.loc["LI-001_0_C", "daño_fisico"] == "PHYSICAL_OK"
    assert informe["no_sin_defecto"] == [{"lechuga": "LI-001", "estado": "0"}]


def test_errores_de_nombres_y_de_la_hoja_se_juntan_en_uno(ingesta_sintetica, fila_hoja, tmp_path):
    ruta = ingesta_sintetica(
        [*_cuatro("LI-001", "0"), *_cuatro("LB-001", "6"), "foto_rara.jpg"],
        [
            fila_hoja("LI-001", "Golpe muy fuerte"),
            fila_hoja("LB-001", "Golpe", danadas="bastantes"),
        ],
    )
    with pytest.raises(ValueError) as error:
        _ingerir(ruta, tmp_path)
    mensaje = str(error.value)
    assert "foto_rara.jpg" in mensaje
    assert "Golpe muy fuerte" in mensaje
    assert "bastantes" in mensaje


def test_variedad_distinta_del_prefijo_da_error(ingesta_sintetica, fila_hoja, tmp_path):
    ruta = ingesta_sintetica(_cuatro("LI-001", "0"), [fila_hoja("LI-001", "0", variedad="Batavia")])
    with pytest.raises(ValueError, match="Batavia.*iceberg"):
        _ingerir(ruta, tmp_path)


def test_lechuga_sin_conservacion_da_error(ingesta_sintetica, fila_hoja, tmp_path):
    ruta = ingesta_sintetica(_cuatro("LI-003", "0"), [fila_hoja("LI-003", "0")])
    with pytest.raises(ValueError, match="LI-003.*conservacion"):
        _ingerir(ruta, tmp_path)


def test_par_lechuga_estado_repetido_en_la_hoja_da_error(ingesta_sintetica, fila_hoja, tmp_path):
    ruta = ingesta_sintetica(
        _cuatro("LI-001", "0"), [fila_hoja("LI-001", "0"), fila_hoja("LI-001", "Tal como viene")]
    )
    with pytest.raises(ValueError, match="filas 2 y 3"):
        _ingerir(ruta, tmp_path)


def test_dispositivo_vacio_avisa(ingesta_sintetica, fila_hoja, tmp_path):
    ruta = ingesta_sintetica(
        _cuatro("LI-001", "0"), [fila_hoja("LI-001", "0")], cambios={"dispositivo": ""}
    )
    informe = _ingerir(ruta, tmp_path)
    assert any("dispositivo" in aviso for aviso in informe["avisos"])


# --- Excepciones --------------------------------------------------------------------------


def _con_excepciones(ingesta_sintetica, fila_hoja, csv: str, **opciones) -> Path:
    return ingesta_sintetica(
        [*_cuatro("LB-001", "0"), *_cuatro("LB-001", "3"), *_cuatro("LB-001", "24h")],
        [fila_hoja("LB-001", "0"), fila_hoja("LB-001", "3"), fila_hoja("LB-001", "24h")],
        excepciones=csv,
        **opciones,
    )


def test_excepcion_sin_vista_cambia_las_cuatro_vistas(ingesta_sintetica, fila_hoja, tmp_path):
    csv = "lechuga,estado,grupo,etiqueta,vista\nLB-001,3,corte,UNCLEAN_CUT,\n"
    informe = _ingerir(_con_excepciones(ingesta_sintetica, fila_hoja, csv), tmp_path)

    df = _metadata(tmp_path).set_index("image_id")
    # También en C, F y G, donde el corte no se ve: la excepción gana sobre la visibilidad.
    assert set(df.loc[[f"LB-001_3_{v}" for v in VISTAS], "corte"]) == {"UNCLEAN_CUT"}
    assert informe["excepciones"] == 1


def test_excepcion_con_vista_cambia_solo_esa(ingesta_sintetica, fila_hoja, tmp_path):
    csv = "lechuga,estado,grupo,etiqueta,vista\nLB-001,3,hojas_ext,OUTER_DAMAGE_MAJOR,B\n"
    _ingerir(_con_excepciones(ingesta_sintetica, fila_hoja, csv), tmp_path)

    df = _metadata(tmp_path).set_index("image_id")
    assert df.loc["LB-001_3_B", "hojas_ext"] == "OUTER_DAMAGE_MAJOR"
    assert df.loc["LB-001_3_C", "hojas_ext"] == "OUTER_LEAVES_OK"


def test_excepcion_no_visible_y_estado_en_texto(ingesta_sintetica, fila_hoja, tmp_path):
    csv = "lechuga,estado,grupo,etiqueta\nLB-001,Corte bueno sin limpiar,color,NO_VISIBLE\n"
    _ingerir(_con_excepciones(ingesta_sintetica, fila_hoja, csv), tmp_path)

    df = _metadata(tmp_path).set_index("image_id")
    assert set(df.loc[[f"LB-001_3_{v}" for v in VISTAS], "color"]) == {NO_VISIBLE}


def test_excepcion_del_estado_0_no_se_hereda_en_24h(ingesta_sintetica, fila_hoja, tmp_path):
    csv = "lechuga,estado,grupo,etiqueta\nLB-001,0,corte,CUT_OK\n"
    _ingerir(_con_excepciones(ingesta_sintetica, fila_hoja, csv), tmp_path)

    df = _metadata(tmp_path).set_index("image_id")
    assert df.loc["LB-001_0_B", "corte"] == "CUT_OK"
    assert df.loc["LB-001_24h_B", "corte"] == "ROOT_REMAINS"  # sigue la regla por variedad


def test_excepciones_con_bom_de_excel(ingesta_sintetica, fila_hoja, tmp_path):
    csv = "lechuga,estado,grupo,etiqueta\nLB-001,3,daño_fisico,TORN_LEAF\n"
    ruta = _con_excepciones(ingesta_sintetica, fila_hoja, csv, codificacion="utf-8-sig")
    _ingerir(ruta, tmp_path)
    assert _metadata(tmp_path).set_index("image_id").loc["LB-001_3_C", "daño_fisico"] == "TORN_LEAF"


@pytest.mark.parametrize(
    "lineas, mensaje",
    [
        ("LB-001,3,cortes,CUT_OK,", "cortes"),
        ("LB-001,3,corte,TRIM_OK,", "TRIM_OK"),
        ("LB-009,3,corte,CUT_OK,", "LB-009"),  # lechuga y estado sin fotos
        ("LB-001,5,corte,CUT_OK,", "LB-001.*5"),
        ("LB-001,3,corte,CUT_OK,X", "'X'"),
        ("LB-001,3,corte,CUT_OK,\nLB-001,3,corte,UNCLEAN_CUT,", "repetida"),
        ("LB-001,3,corte,CUT_OK,\nLB-001,3,corte,UNCLEAN_CUT,B", "repetida"),  # se solapan
    ],
)
def test_excepciones_no_validas_dan_error(ingesta_sintetica, fila_hoja, tmp_path, lineas, mensaje):
    csv = f"lechuga,estado,grupo,etiqueta,vista\n{lineas}\n"
    with pytest.raises(ValueError, match=mensaje):
        _ingerir(_con_excepciones(ingesta_sintetica, fila_hoja, csv), tmp_path)


def test_excepcion_en_una_vista_sin_foto_da_error(ingesta_sintetica, fila_hoja, tmp_path):
    ruta = ingesta_sintetica(
        _cuatro("LB-001", "4", "FGB"),
        [fila_hoja("LB-001", "4")],
        excepciones="lechuga,estado,grupo,etiqueta,vista\nLB-001,4,color,NO_VISIBLE,C\n",
    )
    with pytest.raises(ValueError, match="LB-001_4_C"):
        _ingerir(ruta, tmp_path)


# --- Fotos de salida: orientación y metadatos ---------------------------------------------


def test_orientacion_exif_se_aplica_y_se_quitan_los_metadatos(
    ingesta_sintetica, fila_hoja, jpg_sintetico, tmp_path
):
    ruta = ingesta_sintetica(_cuatro("LI-001", "0"), [fila_hoja("LI-001", "0")])
    # Foto apaisada de 40x20 (mitad izquierda roja) con EXIF "girar 90º" y un comentario.
    jpg_sintetico(
        tmp_path / "raw" / "fotos" / "LI-001_0_C.jpg", 40, 20, orientacion=6, comentario=b"SECRETO"
    )

    _ingerir(ruta, tmp_path)

    datos = (tmp_path / "processed" / "propio" / "LI-001_0_C.jpg").read_bytes()
    assert b"Exif" not in datos
    assert b"SECRETO" not in datos
    imagen = cv2.imdecode(np.frombuffer(datos, np.uint8), cv2.IMREAD_COLOR)
    assert imagen.shape == (40, 20, 3)  # ya girada: ahora es vertical
    rojo, azul = imagen[5, 10], imagen[34, 10]  # la izquierda queda arriba
    assert rojo[2] > 200 and rojo[0] < 60
    assert azul[0] > 200 and azul[2] < 60


# --- metadata.csv -------------------------------------------------------------------------


def test_filas_de_metadata(ingesta_sintetica, fila_hoja, tmp_path):
    ruta = ingesta_sintetica(
        [*_cuatro("LI-002", "0"), *_cuatro("LI-002", "24h")],
        [
            fila_hoja("LI-002", "Tal como viene"),
            fila_hoja("LI-002", "24 h", amarillas="1-2", notas="hojas algo blandas"),
        ],
    )

    _ingerir(ruta, tmp_path)

    df = _metadata(tmp_path)
    assert list(df.columns) == list(METADATA_COLUMNS)
    assert validate_metadata(df) == []
    assert len(df) == 8
    fila = df.set_index("image_id").loc["LI-002_24h_C"]
    assert fila["fuente"] == "propio"
    assert fila["licencia"] == "propia (titularidad por confirmar)"
    assert fila["sesion"] == "piloto"
    assert fila["lechuga_id"] == "LI-002"
    assert fila["vista"] == "cenital"
    assert fila["variedad"] == "iceberg"
    assert fila["estado_seq"] == "0"
    assert fila["horas_desde_corte"] == "24"
    assert fila["conservacion"] == "ambiente"
    assert fila["temperatura_c"] == ""
    assert fila["intervencion"] == "24 h; notas: hojas algo blandas"
    assert fila["etiqueta_original"] == "24 h"
    assert fila["dispositivo"] == "iPhone XX"
    assert fila["archivo_original"] == "LI-002_24h_C.jpg"
    assert fila["color"] == "COLOR_FRESH_OK"  # "1-2" cuenta como 2
    base = df.set_index("image_id").loc["LI-002_0_B"]
    assert (base["estado_seq"], base["horas_desde_corte"]) == ("0", "")
    assert base["intervencion"] == "Tal como viene"
    for _, fila in df.iterrows():
        assert fila["salida_global"] == aggregate({g: fila[g] for g in GROUPS}, TABLA).level


def test_fusion_idempotente_que_conserva_otras_fuentes_y_sesiones(
    ingesta_sintetica, fila_hoja, tmp_path
):
    ruta = ingesta_sintetica(_cuatro("LI-001", "0"), [fila_hoja("LI-001", "0")])
    otra_sesion = _fila_publica(
        "s0_LI-001", fuente="propio", licencia="propia", sesion="s0", lechuga_id="LI-001"
    )
    csv = _escribir_metadata(tmp_path, [_fila_publica("giovi_0001"), otra_sesion])

    informe = _ingerir(ruta, tmp_path)
    primera = csv.read_bytes()
    _ingerir(ruta, tmp_path)

    assert csv.read_bytes() == primera  # la segunda ingesta no cambia nada
    df = _metadata(tmp_path)
    assert list(df["image_id"]) == [
        "giovi_0001", "s0_LI-001", "LI-001_0_B", "LI-001_0_C", "LI-001_0_F", "LI-001_0_G",
    ]  # fmt: skip
    assert any("LI-001" in a and "s0" in a for a in informe["avisos"])


def test_csv_existente_no_valido_no_se_escribe_nada(ingesta_sintetica, fila_hoja, tmp_path):
    ruta = ingesta_sintetica(_cuatro("LI-001", "0"), [fila_hoja("LI-001", "0")])
    csv = _escribir_metadata(tmp_path, [_fila_publica("giovi_0001", corte="MAL")])
    antes = csv.read_bytes()

    with pytest.raises(ValueError, match="MAL"):
        _ingerir(ruta, tmp_path)

    assert csv.read_bytes() == antes
    assert not (tmp_path / "processed" / "propio").exists()


# --- Carpeta de salida --------------------------------------------------------------------


def test_salida_con_archivo_ajeno_se_rechaza_sin_borrar(ingesta_sintetica, fila_hoja, tmp_path):
    ruta = ingesta_sintetica(_cuatro("LI-001", "0"), [fila_hoja("LI-001", "0")])
    salida = tmp_path / "processed" / "propio"
    salida.mkdir(parents=True)
    (salida / "notas.txt").write_text("mío", encoding="utf-8")
    (salida / "LI-001_0_C.jpg").write_bytes(b"viejo")

    with pytest.raises(ValueError, match="notas.txt"):
        _ingerir(ruta, tmp_path)

    assert (salida / "notas.txt").is_file()
    assert (salida / "LI-001_0_C.jpg").read_bytes() == b"viejo"
    assert not (tmp_path / "data" / "metadata.csv").exists()


def test_reingesta_quita_las_fotos_que_ya_no_estan(ingesta_sintetica, fila_hoja, tmp_path):
    ruta = ingesta_sintetica(_cuatro("LI-001", "0"), [fila_hoja("LI-001", "0")])
    _ingerir(ruta, tmp_path)
    (tmp_path / "raw" / "fotos" / "LI-001_0_C.jpg").unlink()

    _ingerir(ruta, tmp_path)

    salida = tmp_path / "processed" / "propio"
    assert not (salida / "LI-001_0_C.jpg").exists()
    assert len(list(salida.glob("*.jpg"))) == 3
    assert len(_metadata(tmp_path)) == 3


def test_comprobar_no_escribe_nada(ingesta_sintetica, fila_hoja, tmp_path):
    ruta = ingesta_sintetica(_cuatro("LI-001", "0"), [fila_hoja("LI-001", "0")])

    informe = _ingerir(ruta, tmp_path, comprobar=True)

    assert informe["fotos"] == 4
    assert not (tmp_path / "processed").exists()
    assert not (tmp_path / "data").exists()


# --- Informe ------------------------------------------------------------------------------


def test_informe_cuenta_fotos_y_etiquetas(ingesta_sintetica, fila_hoja, tmp_path):
    ruta = ingesta_sintetica(
        [*_cuatro("LI-001", "0"), *_cuatro("LB-001", "5", "CFG")],
        [fila_hoja("LI-001", "0"), fila_hoja("LB-001", "5", danadas=4)],
    )

    informe = _ingerir(ruta, tmp_path)

    assert informe["fotos"] == 7 and informe["lechugas"] == 2
    assert informe["fotos_sin_fila"] == []
    assert informe["fotos_por_lechuga_estado"] == [
        {"lechuga": "LB-001", "estado": "5", "fotos": 3, "vistas": ["C", "F", "G"]},
        {"lechuga": "LI-001", "estado": "0", "fotos": 4, "vistas": ["C", "F", "G", "B"]},
    ]
    corte = informe["etiquetas"]["corte"]
    assert corte["CUT_OK"] == 1 and corte[NO_VISIBLE] == 6 and corte["ROOT_REMAINS"] == 0
    assert informe["etiquetas"]["hojas_ext"]["OUTER_DAMAGE_LIGHT"] == 3
    assert sum(informe["salida_global"].values()) == 7
    guardado = json.loads((tmp_path / "processed" / "propio" / "informe.json").read_text("utf-8"))
    assert guardado == informe


def test_sin_rutas_absolutas_en_las_salidas(ingesta_sintetica, fila_hoja, tmp_path):
    ruta = ingesta_sintetica(
        [*_cuatro("LI-001", "0"), "LI-001_24h_C.jpg"],
        [fila_hoja("LI-001", "0"), fila_hoja("LI-001", "24h"), fila_hoja("LI-001", "48h")],
    )

    informe = _ingerir(ruta, tmp_path)

    assert informe["config"] == "configs/propio.yaml"
    textos = [
        (tmp_path / "processed" / "propio" / "informe.json").read_text("utf-8"),
        (tmp_path / "data" / "metadata.csv").read_text("utf-8"),
    ]
    for texto in textos:
        for absoluta in (str(tmp_path), tmp_path.as_posix(), str(tmp_path).replace("\\", "\\\\")):
            assert absoluta not in texto
