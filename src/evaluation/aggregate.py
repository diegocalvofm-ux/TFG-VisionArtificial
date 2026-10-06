"""Regla de agregación: de las etiquetas de los seis grupos al nivel global de la lechuga."""

from collections.abc import Mapping
from typing import NamedTuple

from src.data.schema import GLOBAL_LEVELS, GROUPS, NO_VISIBLE, is_blank

# Nivel de una etiqueta que el manual no fija todavía (falta validar con tutor y experto).
PENDIENTE = "PENDIENTE"

# GLOBAL_LEVELS va de mejor a peor: ACCEPT, REVIEW, NON-CONFORMING (lo protege una prueba
# del esquema). La gravedad de un nivel es, por tanto, su posición en esa tupla.
_REVIEW = GLOBAL_LEVELS.index("REVIEW")


class Aggregation(NamedTuple):
    """Resultado de agregar: el nivel global y las etiquetas provisionales presentes."""

    level: str
    provisional_labels: list[str]


def _validar_grupos(grupos: Mapping[str, str]) -> None:
    """Lanza ValueError si faltan grupos, sobran grupos o hay una etiqueta que no es del grupo."""
    faltan = [grupo for grupo in GROUPS if grupo not in grupos]
    if faltan:
        raise ValueError(f"Faltan grupos en la entrada: {', '.join(faltan)}")
    sobran = [grupo for grupo in grupos if grupo not in GROUPS]
    if sobran:
        raise ValueError(
            f"Grupos que no existen: {', '.join(map(str, sobran))} "
            f"(los válidos son: {', '.join(GROUPS)})"
        )
    for grupo, etiquetas in GROUPS.items():
        etiqueta = grupos[grupo]
        if is_blank(etiqueta) or (etiqueta != NO_VISIBLE and etiqueta not in etiquetas):
            raise ValueError(
                f"La etiqueta {etiqueta!r} no pertenece al grupo '{grupo}' "
                f"(válidas: {', '.join((*etiquetas, NO_VISIBLE))})"
            )


def _nivel_en_tabla(grupo: str, etiqueta: str, tabla: Mapping[str, Mapping[str, str]]) -> str:
    """Busca el nivel de una etiqueta en la tabla; ValueError si falta o no es un nivel válido."""
    try:
        nivel = tabla[grupo][etiqueta]
    except KeyError:
        raise ValueError(
            f"La tabla de salida global no tiene nivel para '{etiqueta}' en el grupo '{grupo}'"
        ) from None
    if nivel != PENDIENTE and nivel not in GLOBAL_LEVELS:
        raise ValueError(
            f"Nivel {nivel!r} no válido para '{etiqueta}' en la tabla "
            f"(válidos: {', '.join((*GLOBAL_LEVELS, PENDIENTE))})"
        )
    return nivel


def aggregate(
    grupos: Mapping[str, str], tabla: Mapping[str, Mapping[str, str]]
) -> Aggregation:
    """Calcula el nivel global (ACCEPT / REVIEW / NON-CONFORMING) de una lechuga.

    En lenguaje llano: cada uno de los seis grupos aporta una etiqueta (o ``NO_VISIBLE``) y
    la tabla dice qué nivel merece cada etiqueta. El resultado es el PEOR nivel de los seis,
    con estas particularidades:

    - Una etiqueta ``PENDIENTE`` (nivel aún sin validar) cuenta como REVIEW y se anota en
      ``provisional_labels``. Se anotan todas las presentes, aunque no decidan el resultado.
    - ``NO_VISIBLE`` también cuenta como REVIEW: si algún grupo no se ve, el resultado no
      puede ser mejor que REVIEW (y si otra etiqueta da NON-CONFORMING, sigue siéndolo).
      Esta lectura del "máximo REVIEW" del manual está pendiente de confirmar con el tutor.
    - Por tanto, ACCEPT solo sale si los seis grupos son visibles y todos ACCEPT; y si
      ninguno es visible el resultado es REVIEW.

    ``grupos`` es ``{grupo: etiqueta}`` con los seis grupos de ``GROUPS``; ``tabla`` es la de
    ``configs/salida_global.yaml`` (``{grupo: {etiqueta: nivel}}``). No se modifican.
    ``provisional_labels`` sale en el orden de ``GROUPS``, no en el de ``grupos``.

    Errores (ValueError): falta un grupo, sobra un grupo, una etiqueta no es de su grupo, o
    la tabla no tiene nivel (o tiene un nivel inválido) para una etiqueta usada.
    """
    _validar_grupos(grupos)

    peor = 0  # gravedad del peor nivel visto; 0 = ACCEPT
    provisionales: list[str] = []
    for grupo in GROUPS:  # en el orden de GROUPS, para un resultado determinista
        etiqueta = grupos[grupo]
        if etiqueta == NO_VISIBLE:
            gravedad = _REVIEW
        else:
            nivel = _nivel_en_tabla(grupo, etiqueta, tabla)
            if nivel == PENDIENTE:
                gravedad = _REVIEW
                provisionales.append(etiqueta)
            else:
                gravedad = GLOBAL_LEVELS.index(nivel)
        peor = max(peor, gravedad)

    return Aggregation(level=GLOBAL_LEVELS[peor], provisional_labels=provisionales)
