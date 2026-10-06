"""Pruebas de src/config.py: lectura de configuraciones YAML."""

from pathlib import Path

import pytest

from src.config import load_config

RAIZ = Path(__file__).resolve().parents[2]


def test_lee_yaml_valido_con_str_y_path(tmp_path):
    # Con tildes y tipos mixtos, para comprobar también la lectura en UTF-8.
    archivo = tmp_path / "ejemplo.yaml"
    archivo.write_text(
        "detector: whole\numbral: 0.25\nconsulta: \"lechuga pequeña\"\nlista: [1, 2]\n",
        encoding="utf-8",
    )
    esperado = {
        "detector": "whole",
        "umbral": 0.25,
        "consulta": "lechuga pequeña",
        "lista": [1, 2],
    }

    assert load_config(archivo) == esperado
    assert load_config(str(archivo)) == esperado


def test_archivo_inexistente_da_error_claro(tmp_path):
    ruta = tmp_path / "no_existe.yaml"

    with pytest.raises(FileNotFoundError, match="no_existe.yaml"):
        load_config(ruta)


@pytest.mark.parametrize("contenido", ["", "- uno\n- dos\n"], ids=["vacio", "lista"])
def test_yaml_que_no_es_diccionario_da_error(tmp_path, contenido):
    archivo = tmp_path / "malo.yaml"
    archivo.write_text(contenido, encoding="utf-8")

    with pytest.raises(ValueError, match="diccionario"):
        load_config(archivo)


def test_config_v0_real_tiene_las_claves_esperadas():
    # Protege a las tareas 3 y 4, que leerán estas claves.
    cfg = load_config(RAIZ / "configs" / "v0.yaml")

    assert cfg["detector"] in {"whole", "owlvit"}
    assert isinstance(cfg["owlvit"]["umbral"], float)
    assert isinstance(cfg["owlvit"]["consulta"], str) and cfg["owlvit"]["consulta"]
