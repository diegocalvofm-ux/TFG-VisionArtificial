"""Pruebas de los detectores: tipos, detector "whole" y registro."""

import dataclasses

import pytest

from src.pipeline.detectors import DETECTORS, crear_detector, nombres_detectores
from src.pipeline.detectors.whole import WholeDetector
from src.pipeline.types import Detection


def test_detection_tiene_etiqueta_por_defecto_y_es_inmutable():
    deteccion = Detection(box=(0, 0, 10, 20), score=0.5)

    assert deteccion.label == "lettuce"
    with pytest.raises(dataclasses.FrozenInstanceError):
        deteccion.score = 1.0  # type: ignore[misc]


@pytest.mark.parametrize(("alto", "ancho"), [(48, 64), (100, 30)], ids=["apaisada", "vertical"])
def test_caso1_whole_devuelve_una_caja_con_el_tamano_de_la_imagen(crear_imagen, alto, ancho):
    detecciones = WholeDetector().detect(crear_imagen(alto, ancho))

    assert len(detecciones) == 1
    assert detecciones[0].box == (0, 0, ancho, alto)  # (x1, y1, x2, y2): ancho y alto reales
    assert detecciones[0].score == 1.0


def test_el_registro_incluye_whole_y_lo_crea():
    assert "whole" in nombres_detectores()

    assert isinstance(crear_detector("whole", {}), WholeDetector)


def test_caso9_detector_desconocido_da_un_error_claro_que_lista_los_disponibles():
    with pytest.raises(ValueError, match="nope") as error:
        crear_detector("nope", {})

    for disponible in nombres_detectores():
        assert disponible in str(error.value)
    assert "whole" in str(error.value)


def test_la_fabrica_recibe_la_configuracion(monkeypatch):
    recibido = {}

    def fabrica(cfg):
        recibido.update(cfg)
        return WholeDetector()

    monkeypatch.setitem(DETECTORS, "falso", fabrica)

    crear_detector("falso", {"owlvit": {"umbral": 0.1}})

    assert recibido == {"owlvit": {"umbral": 0.1}}
