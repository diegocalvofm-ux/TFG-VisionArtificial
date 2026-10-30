"""Pruebas de src/data/importar_publicos.py: la línea de comandos (listar, importar, revisar)."""

import json
from pathlib import Path

from src.data.importar_publicos import main

RECT = [[2, 2, 10, 2, 10, 8, 2, 8]]


def _arbol(carpeta: Path) -> list[str]:
    """Lista ordenada de todo lo que hay bajo una carpeta (nombres y tamaños)."""
    return sorted(f"{p.relative_to(carpeta)}:{p.stat().st_size}" for p in carpeta.rglob("*"))


def test_listar_muestra_clases_y_conteos_sin_escribir_nada(
    zip_coco_sintetico, tmp_path, capsys, monkeypatch
):
    nombres = [f"img_{n:02d}.jpg" for n in range(30)]
    ruta = zip_coco_sintetico(
        {
            "train": [{"nombre": n, "anotaciones": [("lechuga", RECT)] * 2} for n in nombres[:25]],
            "valid": [{"nombre": n, "anotaciones": [("kern", RECT)]} for n in nombres[25:]],
        }
    )
    monkeypatch.chdir(tmp_path)
    antes = _arbol(tmp_path)

    codigo = main(["listar", "--zip", str(ruta)])

    salida = capsys.readouterr().out
    assert codigo == 0
    assert _arbol(tmp_path) == antes  # no escribe nada
    assert "lechuga: 50" in salida  # 25 imágenes x 2 anotaciones
    assert "kern: 5" in salida
    assert "padre: 0" in salida  # la categoría sin anotaciones se ve, con 0
    assert "train: 25" in salida and "valid: 5" in salida
    assert sum(1 for n in nombres if n in salida) == 20  # solo 20 ejemplos


def test_importar_y_revisar_de_extremo_a_extremo(
    fuente_sintetica, etapa_sintetica, tmp_path, capsys
):
    carpetas = {
        "train": [
            {"nombre": f"g{g:02d}_{k}.jpg", "anotaciones": [("lechuga", RECT)]}
            for g in range(12)
            for k in range(2)
        ]
    }
    fuente_sintetica(
        "pub",
        carpetas,
        {"lechuga": "lechuga"},
        papel="split",
        fracciones={"train": 0.5, "valid": 0.25, "test": 0.25},
        grupo_regex=r"^(g\d+)_",
    )
    etapa = etapa_sintetica(["pub"])

    codigo = main(
        [
            "importar",
            "--etapa", str(etapa),
            "--raiz", str(tmp_path),
            "--metadata", str(tmp_path / "data" / "metadata.csv"),
        ]
    )  # fmt: skip

    assert codigo == 0
    assert (tmp_path / "data" / "metadata.csv").is_file()
    informe = json.loads(
        (tmp_path / "data/processed/etapa1_v1/informe.json").read_text(encoding="utf-8")
    )
    assert informe["fuentes"]["pub"]["particiones"]["train"]["imagenes"] > 0
    assert "pub" in capsys.readouterr().out  # el informe también va a la consola

    destino = tmp_path / "runs" / "checks"
    codigo = main(
        [
            "revisar",
            "--salida", str(tmp_path / "data/processed/etapa1_v1"),
            "--particion", "train",
            "--n", "3",
            "--semilla", "0",
            "--destino", str(destino),
        ]
    )  # fmt: skip

    assert codigo == 0
    assert len(list((destino / "train").glob("*.jpg"))) == 3


def test_error_de_clase_sin_mapeo_sale_con_codigo_1_y_mensaje(
    fuente_sintetica, etapa_sintetica, tmp_path, capsys
):
    fuente_sintetica(
        "pub",
        {"train": [{"nombre": "a.jpg", "anotaciones": [("kern", RECT)]}]},
        {"lechuga": "lechuga"},
    )
    etapa = etapa_sintetica(["pub"])

    codigo = main(["importar", "--etapa", str(etapa), "--raiz", str(tmp_path)])

    assert codigo == 1
    assert "kern" in capsys.readouterr().err
