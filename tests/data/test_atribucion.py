"""Pruebas de src/data/atribucion.py: ATTRIBUTION.md generado desde configs/datasets/."""

from pathlib import Path

import pytest
import yaml

from src.data.atribucion import PENDIENTE, fuentes_publicas, generar, main

RAIZ_REPO = Path(__file__).resolve().parents[2]
CARPETA_REAL = RAIZ_REPO / "configs" / "datasets"


def _descriptor(carpeta: Path, nombre: str, **campos) -> None:
    datos = {
        "fuente": nombre,
        "nombre": f"dataset {nombre}",
        "licencia": "MIT",
        "zip": f"data/raw/{nombre}.zip",
        "autor": "alguien",
        "url": f"https://ejemplo.org/{nombre}",
        "fecha_descarga": "2026-11-16",
    }
    datos.update(campos)
    carpeta.mkdir(parents=True, exist_ok=True)
    (carpeta / f"{nombre}.yaml").write_text(yaml.safe_dump(datos, allow_unicode=True),
                                            encoding="utf-8")  # fmt: skip


def test_attribution_md_esta_al_dia_con_los_yaml():
    esperado = generar(CARPETA_REAL)
    actual = (RAIZ_REPO / "ATTRIBUTION.md").read_text(encoding="utf-8")
    assert actual == esperado, "ATTRIBUTION.md no está al día: python -m src.data.atribucion"


def test_estan_las_cinco_fuentes_publicas_con_sus_datos():
    fuentes = {f["fuente"]: f for f in fuentes_publicas(CARPETA_REAL)}
    assert set(fuentes) == {"giovi", "svodk", "tipburn", "yssp", "phs"}
    assert fuentes["yssp"]["licencia"] == "CC BY 4.0"
    assert fuentes["yssp"]["autor"] == "yssp"
    assert fuentes["yssp"]["url"] == "https://universe.roboflow.com/yssp/disease-lettuce-h69zb"
    assert fuentes["phs"]["licencia"] == "MIT"
    assert fuentes["phs"]["autor"] == "PHS"
    assert fuentes["phs"]["url"] == "https://universe.roboflow.com/phs/lettuce_disease"
    assert fuentes["tipburn"]["nombre"] == "Lettuce_Segmentation"
    for nombre in ("giovi", "svodk", "tipburn"):  # los rellena el autor más adelante
        assert fuentes[nombre]["autor"] == PENDIENTE
        assert fuentes[nombre]["url"] == PENDIENTE
    for fuente in fuentes.values():
        assert fuente["fecha_descarga"] in ("2026-10-27", "2026-11-16")


def test_campo_nulo_sale_como_pendiente_y_los_yaml_sin_zip_se_ignoran(tmp_path):
    _descriptor(tmp_path, "uno", autor=None, url=None)
    (tmp_path / "etapa.yaml").write_text("fuentes: [uno]\n", encoding="utf-8")  # sin zip

    texto = generar(tmp_path)

    assert "| uno | dataset uno | PENDIENTE | PENDIENTE | MIT | 2026-11-16 |" in texto
    assert "etapa" not in texto


def test_fuente_sin_licencia_da_error(tmp_path):
    _descriptor(tmp_path, "uno", licencia="")
    with pytest.raises(ValueError, match="licencia"):
        generar(tmp_path)


def test_main_escribe_y_comprobar_detecta_un_archivo_desfasado(tmp_path, capsys):
    carpeta, salida = tmp_path / "configs", tmp_path / "ATTRIBUTION.md"
    _descriptor(carpeta, "uno")
    salida.write_text("viejo\n", encoding="utf-8")
    argumentos = ["--carpeta", str(carpeta), "--salida", str(salida)]

    assert main([*argumentos, "--comprobar"]) == 1
    assert "no está al día" in capsys.readouterr().err
    assert main(argumentos) == 0
    assert salida.read_text(encoding="utf-8") == generar(carpeta)
    assert main([*argumentos, "--comprobar"]) == 0
