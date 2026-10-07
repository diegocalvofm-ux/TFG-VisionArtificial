"""Pruebas del asesor nulo (v0: todavía no hay modelo de calidad)."""

from src.data.schema import GROUPS, NO_VISIBLE
from src.pipeline.assessors.null import NullAssessor
from src.pipeline.types import Detection


def test_caso2_el_asesor_nulo_devuelve_los_seis_grupos_como_no_visible(crear_imagen):
    imagen = crear_imagen()

    grupos = NullAssessor().assess(imagen, Detection(box=(0, 0, 64, 48), score=1.0))

    assert list(grupos) == list(GROUPS)  # los seis, en el orden del esquema
    assert len(grupos) == 6
    assert set(grupos.values()) == {NO_VISIBLE}
    assert "daño_fisico" in grupos and "daño_bio" in grupos
