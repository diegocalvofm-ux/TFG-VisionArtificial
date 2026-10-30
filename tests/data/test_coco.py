"""Pruebas de src/data/coco.py: lectura de ZIP COCO y geometría de polígonos."""

import json
import zipfile

import pytest

from src.data.coco import (
    area_poligonos,
    bbox_desde_poligonos,
    clases_con_anotaciones,
    escribir_coco,
    leer_anotaciones_zip,
    poligono_valido,
)

# Rectángulo de (2, 2) a (10, 8): caja [2, 2, 8, 6] y área 48.
POLIGONO_RECTANGULO = [[2, 2, 10, 2, 10, 8, 2, 8]]


def test_bbox_de_un_cuadrado():
    assert bbox_desde_poligonos([[2, 2, 10, 2, 10, 8, 2, 8]]) == (2, 2, 8, 6)


def test_bbox_de_un_triangulo():
    assert bbox_desde_poligonos([[0, 0, 4, 0, 2, 6]]) == (0, 0, 4, 6)


def test_bbox_de_varios_poligonos_los_abarca_a_todos():
    poligonos = [[0, 0, 2, 0, 2, 2], [10, 10, 12, 10, 12, 15]]
    assert bbox_desde_poligonos(poligonos) == (0, 0, 12, 15)


def test_area_de_un_rectangulo_y_de_dos_poligonos():
    assert area_poligonos([[2, 2, 10, 2, 10, 8, 2, 8]]) == 48
    # triángulo de base 4 y altura 6 (12) + cuadrado 2x2 (4)
    assert area_poligonos([[0, 0, 4, 0, 2, 6], [0, 0, 2, 0, 2, 2, 0, 2]]) == 16


def test_poligono_degenerado_no_es_valido():
    assert poligono_valido([0, 0, 4, 0, 2, 6])
    assert not poligono_valido([0, 0, 4, 0])  # solo dos puntos
    assert not poligono_valido([0, 0, 1, 1, 2, 2])  # puntos alineados: área 0
    assert not poligono_valido([0, 0, 4, 0, 2])  # número impar de coordenadas


def test_clases_con_anotaciones_cuenta_por_nombre_y_ignora_la_categoria_padre(
    zip_coco_sintetico,
):
    ruta = zip_coco_sintetico(
        {
            "train": [
                {
                    "nombre": "a.jpg",
                    "anotaciones": [("lechuga", POLIGONO_RECTANGULO)] * 2
                    + [("kern", POLIGONO_RECTANGULO)],
                }
            ]
        }
    )
    with zipfile.ZipFile(ruta) as zf:
        coco = leer_anotaciones_zip(zf)["train"]

    assert dict(clases_con_anotaciones(coco)) == {"lechuga": 2, "kern": 1}  # sin "padre"


def test_leer_anotaciones_zip_sin_json_da_error(tmp_path):
    ruta = tmp_path / "vacio.zip"
    with zipfile.ZipFile(ruta, "w") as zf:
        zf.writestr("train/a.jpg", b"x")
    with zipfile.ZipFile(ruta) as zf, pytest.raises(ValueError, match="_annotations.coco.json"):
        leer_anotaciones_zip(zf)


def test_escribir_coco_guarda_utf8_y_estructura(tmp_path):
    destino = tmp_path / "t.json"
    escribir_coco(
        destino,
        imagenes=[{"id": 1, "file_name": "a.jpg", "width": 20, "height": 20}],
        anotaciones=[{"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 1, 1]}],
        categorias=[{"id": 1, "name": "lechuga"}],
    )
    leido = json.loads(destino.read_text(encoding="utf-8"))
    assert set(leido) == {"images", "annotations", "categories"}
    assert leido["categories"] == [{"id": 1, "name": "lechuga"}]
