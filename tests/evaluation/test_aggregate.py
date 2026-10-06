"""Pruebas de src/evaluation/aggregate.py: regla de agregación de la salida global.

Las expectativas están escritas a mano (a partir del brief), sin usar el código que se prueba.
"""

import copy
import itertools
import random
from pathlib import Path

import pytest

from src.config import load_config
from src.data.schema import GLOBAL_LEVELS, GROUPS, NO_VISIBLE
from src.evaluation.aggregate import PENDIENTE, aggregate

RAIZ = Path(__file__).resolve().parents[2]

TODOS_OK = {
    "corte": "CUT_OK",
    "recorte": "TRIM_OK",
    "hojas_ext": "OUTER_LEAVES_OK",
    "color": "COLOR_FRESH_OK",
    "daño_fisico": "PHYSICAL_OK",
    "daño_bio": "BIO_OK",
}
TODOS_NO_VISIBLE = {grupo: NO_VISIBLE for grupo in TODOS_OK}

# Niveles que el brief asigna de forma explícita (etiqueta -> nivel). Todo lo demás es
# PENDIENTE: no se rellena con criterios de OECD, USDA ni otra fuente sin validar.
NIVELES_EXPLICITOS = {
    "CUT_OK": "ACCEPT",
    "TRIM_OK": "ACCEPT",
    "OUTER_LEAVES_OK": "ACCEPT",
    "COLOR_FRESH_OK": "ACCEPT",
    "PHYSICAL_OK": "ACCEPT",
    "BIO_OK": "ACCEPT",
    "ROOT_REMAINS": "NON-CONFORMING",
    "ROT_DECAY": "NON-CONFORMING",
    "OUTER_DAMAGE_LIGHT": "REVIEW",
    "STEM_TOO_LONG": "REVIEW",
    "TORN_LEAF": "REVIEW",
}
ETIQUETAS_PENDIENTES_DEL_BRIEF = (
    "VISIBLE_PEST",
    "YELLOW_BROWN_LIGHT",
    "YELLOW_BROWN_MAJOR",
    "WILTING",
    "UNCLEAN_CUT",
)

# Opciones de cada grupo para las pruebas exhaustivas: sus etiquetas más NO_VISIBLE.
OPCIONES = {grupo: (*etiquetas, NO_VISIBLE) for grupo, etiquetas in GROUPS.items()}


@pytest.fixture(scope="module")
def tabla():
    return load_config(RAIZ / "configs" / "salida_global.yaml")


def con(base: dict, **cambios) -> dict:
    """Copia de ``base`` con algunos grupos cambiados (por nombre de grupo)."""
    return {**base, **cambios}


def ok_salvo(grupo: str, etiqueta: str) -> dict:
    return con(TODOS_OK, **{grupo: etiqueta})


def no_visible_salvo(grupo: str, etiqueta: str) -> dict:
    return con(TODOS_NO_VISIBLE, **{grupo: etiqueta})


# --- Casos 1 a 10: la regla -------------------------------------------------------------


def test_caso1_los_seis_ok_dan_accept(tabla):
    resultado = aggregate(TODOS_OK, tabla)

    assert resultado.level == "ACCEPT"
    assert resultado.provisional_labels == []


@pytest.mark.parametrize("grupo", list(TODOS_OK))
def test_caso2_un_grupo_no_visible_y_el_resto_ok_da_review(tabla, grupo):
    resultado = aggregate(ok_salvo(grupo, NO_VISIBLE), tabla)

    assert resultado.level == "REVIEW"
    assert resultado.provisional_labels == []


def test_caso3_root_remains_y_el_resto_no_visible_da_non_conforming(tabla):
    resultado = aggregate(no_visible_salvo("corte", "ROOT_REMAINS"), tabla)

    assert resultado.level == "NON-CONFORMING"
    assert resultado.provisional_labels == []


def test_caso4_rot_decay_y_el_resto_ok_da_non_conforming(tabla):
    resultado = aggregate(ok_salvo("daño_bio", "ROT_DECAY"), tabla)

    assert resultado.level == "NON-CONFORMING"
    assert resultado.provisional_labels == []


@pytest.mark.parametrize(
    ("grupo", "etiqueta"),
    [
        ("hojas_ext", "OUTER_DAMAGE_LIGHT"),
        ("corte", "STEM_TOO_LONG"),
        ("daño_fisico", "TORN_LEAF"),
    ],
)
def test_caso5_etiquetas_review_con_el_resto_ok_dan_review(tabla, grupo, etiqueta):
    resultado = aggregate(ok_salvo(grupo, etiqueta), tabla)

    assert resultado.level == "REVIEW"
    assert resultado.provisional_labels == []


def test_caso6_ningun_grupo_visible_da_review(tabla):
    resultado = aggregate(TODOS_NO_VISIBLE, tabla)

    assert resultado.level == "REVIEW"
    assert resultado.provisional_labels == []


def test_caso7_wilting_con_el_resto_ok_da_review_provisional(tabla):
    resultado = aggregate(ok_salvo("color", "WILTING"), tabla)

    assert resultado.level == "REVIEW"
    assert resultado.provisional_labels == ["WILTING"]


def test_caso8_visible_pest_da_review_provisional(tabla):
    resultado = aggregate(ok_salvo("daño_bio", "VISIBLE_PEST"), tabla)

    assert resultado.level == "REVIEW"
    assert resultado.provisional_labels == ["VISIBLE_PEST"]


def test_caso9_root_remains_y_outer_damage_light_dan_non_conforming(tabla):
    grupos = con(TODOS_OK, corte="ROOT_REMAINS", hojas_ext="OUTER_DAMAGE_LIGHT")

    assert aggregate(grupos, tabla).level == "NON-CONFORMING"


def test_caso10_root_remains_y_wilting_dan_non_conforming_y_wilting_sigue_provisional(tabla):
    grupos = con(TODOS_OK, corte="ROOT_REMAINS", color="WILTING")

    resultado = aggregate(grupos, tabla)

    assert resultado.level == "NON-CONFORMING"
    assert resultado.provisional_labels == ["WILTING"]


def test_pendiente_con_el_resto_no_visible_da_review_provisional(tabla):
    resultado = aggregate(no_visible_salvo("corte", "UNCLEAN_CUT"), tabla)

    assert resultado.level == "REVIEW"
    assert resultado.provisional_labels == ["UNCLEAN_CUT"]


def test_provisional_labels_lista_todas_las_pendientes_en_el_orden_de_groups(tabla):
    grupos = con(TODOS_OK, daño_bio="VISIBLE_PEST", color="WILTING", corte="UNCLEAN_CUT")
    grupos_al_reves = dict(reversed(list(grupos.items())))

    resultado = aggregate(grupos_al_reves, tabla)

    assert resultado.provisional_labels == ["UNCLEAN_CUT", "WILTING", "VISIBLE_PEST"]


def test_el_resultado_se_puede_desempaquetar_y_leer_por_nombre(tabla):
    nivel, provisionales = aggregate(ok_salvo("color", "WILTING"), tabla)
    resultado = aggregate(ok_salvo("color", "WILTING"), tabla)

    assert (nivel, provisionales) == (resultado.level, resultado.provisional_labels)
    assert isinstance(resultado.provisional_labels, list)


def test_aggregate_no_modifica_sus_entradas(tabla):
    grupos = con(TODOS_OK, color="WILTING")
    grupos_antes, tabla_antes = copy.deepcopy(grupos), copy.deepcopy(tabla)

    aggregate(grupos, tabla)

    assert grupos == grupos_antes
    assert tabla == tabla_antes


# --- Caso 11: entradas inválidas --------------------------------------------------------


@pytest.mark.parametrize(
    ("grupos", "fragmento"),
    [
        (con(TODOS_OK, corte="NO_EXISTE"), "NO_EXISTE"),  # etiqueta desconocida
        (con(TODOS_OK, corte="BIO_OK"), "BIO_OK"),  # etiqueta de otro grupo
        (con(TODOS_OK, corte=""), "corte"),  # etiqueta vacía
        (con(TODOS_OK, corte=None), "corte"),  # etiqueta nula
        ({g: e for g, e in TODOS_OK.items() if g != "color"}, "color"),  # grupo ausente
        (con(TODOS_OK, otro_grupo="CUT_OK"), "otro_grupo"),  # grupo que no existe
    ],
    ids=["desconocida", "de_otro_grupo", "vacia", "nula", "grupo_ausente", "grupo_inexistente"],
)
def test_caso11_entradas_invalidas_lanzan_value_error(tabla, grupos, fragmento):
    with pytest.raises(ValueError, match=fragmento):
        aggregate(grupos, tabla)


def test_tabla_sin_una_etiqueta_lanza_value_error(tabla):
    rota = copy.deepcopy(tabla)
    del rota["corte"]["CUT_OK"]

    with pytest.raises(ValueError, match="CUT_OK"):
        aggregate(TODOS_OK, rota)


def test_tabla_con_un_nivel_invalido_lanza_value_error(tabla):
    rota = copy.deepcopy(tabla)
    rota["corte"]["CUT_OK"] = "GENIAL"

    with pytest.raises(ValueError, match="GENIAL"):
        aggregate(TODOS_OK, rota)


# --- Caso 12: monotonía -----------------------------------------------------------------


def _gravedad(etiqueta: str, niveles: dict) -> int:
    """Gravedad de una etiqueta: 0 ACCEPT, 1 REVIEW, 2 NON-CONFORMING.

    NO_VISIBLE y PENDIENTE valen como REVIEW (es lo que fuerzan en el resultado).
    """
    nivel = "REVIEW" if etiqueta == NO_VISIBLE else niveles[etiqueta]
    return GLOBAL_LEVELS.index("REVIEW" if nivel == PENDIENTE else nivel)


def _niveles_planos(tabla: dict) -> dict:
    """Aplana la tabla grupo -> etiqueta -> nivel a etiqueta -> nivel."""
    return {e: nivel for etiquetas in tabla.values() for e, nivel in etiquetas.items()}


def test_caso12_ninguna_etiqueta_da_un_resultado_mejor_que_su_propio_nivel(tabla):
    niveles = _niveles_planos(tabla)
    combinaciones = list(itertools.product(*OPCIONES.values()))
    assert len(combinaciones) == 9600  # 5 * 4 * 4 * 5 * 4 * 6

    for combinacion in combinaciones:
        grupos = dict(zip(OPCIONES, combinacion, strict=True))
        graves = [_gravedad(etiqueta, niveles) for etiqueta in combinacion]
        obtenido = GLOBAL_LEVELS.index(aggregate(grupos, tabla).level)

        assert obtenido >= max(graves), grupos  # nunca mejor que la peor etiqueta
        # y es exactamente la peor (así ACCEPT solo sale si los seis son visibles y ACCEPT)
        assert obtenido == max(graves), grupos


def test_caso12_sustituir_una_etiqueta_por_otra_peor_nunca_mejora_el_resultado(tabla):
    niveles = _niveles_planos(tabla)
    azar = random.Random(0)  # semilla fija: la prueba es reproducible
    comprobadas = 0

    for _ in range(300):
        grupos = {grupo: azar.choice(opciones) for grupo, opciones in OPCIONES.items()}
        grupo = azar.choice(list(OPCIONES))
        nueva = azar.choice(OPCIONES[grupo])
        if _gravedad(nueva, niveles) < _gravedad(grupos[grupo], niveles):
            continue  # solo interesa sustituir por una etiqueta igual o más grave
        cambiados = {**grupos, grupo: nueva}

        antes = GLOBAL_LEVELS.index(aggregate(grupos, tabla).level)
        despues = GLOBAL_LEVELS.index(aggregate(cambiados, tabla).level)

        assert despues >= antes, (grupos, cambiados)
        comprobadas += 1

    assert comprobadas > 100  # que la prueba no se quede vacía


# --- Caso 13: el YAML coincide con el esquema -------------------------------------------


def test_caso13_el_yaml_coincide_con_groups(tabla):
    assert set(tabla) == set(GROUPS)
    for grupo, etiquetas in GROUPS.items():
        assert set(tabla[grupo]) == set(etiquetas), f"etiquetas distintas en '{grupo}'"


def test_caso13_los_niveles_del_yaml_son_validos_y_no_aparece_no_visible(tabla):
    permitidos = {*GLOBAL_LEVELS, PENDIENTE}

    for grupo, etiquetas in tabla.items():
        assert NO_VISIBLE not in etiquetas, grupo
        for etiqueta, nivel in etiquetas.items():
            assert nivel in permitidos, f"{grupo}/{etiqueta}: nivel '{nivel}' no válido"


def test_caso13_solo_las_etiquetas_del_brief_tienen_nivel_y_las_demas_son_pendiente(tabla):
    todas = {etiqueta for etiquetas in GROUPS.values() for etiqueta in etiquetas}
    assert set(NIVELES_EXPLICITOS) <= todas  # ninguna errata en la lista del brief

    esperada = {
        grupo: {etiqueta: NIVELES_EXPLICITOS.get(etiqueta, PENDIENTE) for etiqueta in etiquetas}
        for grupo, etiquetas in GROUPS.items()
    }

    assert tabla == esperada


@pytest.mark.parametrize("etiqueta", ETIQUETAS_PENDIENTES_DEL_BRIEF)
def test_caso13_las_etiquetas_ambiguas_siguen_pendientes(tabla, etiqueta):
    assert _niveles_planos(tabla)[etiqueta] == PENDIENTE
