"""Pruebas de la lectura del conjunto COCO con cajas de referencia."""

import json
import warnings

import numpy as np
import pytest

from src.eval.conjunto import NOMBRE_ANOTACIONES, cargar_conjunto_coco


def _carpeta(raiz):
    return raiz / "conjunto" / "valid"


def _editar_json(carpeta, cambiar):
    ruta = carpeta / NOMBRE_ANOTACIONES
    datos = json.loads(ruta.read_text(encoding="utf-8"))
    cambiar(datos)
    ruta.write_text(json.dumps(datos, ensure_ascii=False), encoding="utf-8")


def test_lee_imagenes_cajas_y_tamanos(raiz_sintetica):
    imagenes = cargar_conjunto_coco(_carpeta(raiz_sintetica), "lechuga")
    assert [i.nombre for i in imagenes] == ["a_ñ.png", "b.png", "c.png"]  # con ñ
    assert [(i.ancho, i.alto) for i in imagenes] == [(120, 100)] * 3
    assert all(i.ruta.is_file() for i in imagenes)
    # bbox COCO [x, y, ancho, alto] -> esquinas (x1, y1, x2, y2).
    np.testing.assert_array_equal(imagenes[0].cajas, [[10, 10, 50, 50], [60, 10, 100, 50]])
    np.testing.assert_array_equal(imagenes[1].cajas, [[20, 20, 80, 80]])


def test_una_imagen_sin_cajas_es_valida(raiz_sintetica):
    carpeta = _carpeta(raiz_sintetica)
    _editar_json(carpeta, lambda d: d.update(annotations=[a for a in d["annotations"]
                                                          if a["image_id"] != 3]))  # fmt: skip
    imagenes = cargar_conjunto_coco(carpeta, "lechuga")
    assert imagenes[2].cajas.shape == (0, 4)


def test_sin_archivo_de_anotaciones(tmp_path):
    with pytest.raises(FileNotFoundError, match="_annotations.coco.json"):
        cargar_conjunto_coco(tmp_path, "lechuga")


def test_clase_que_no_esta_en_el_conjunto(raiz_sintetica):
    with pytest.raises(ValueError, match="repollo"):
        cargar_conjunto_coco(_carpeta(raiz_sintetica), "repollo")


def test_imagen_que_falta_en_disco(raiz_sintetica):
    carpeta = _carpeta(raiz_sintetica)
    (carpeta / "b.png").unlink()
    with pytest.raises(FileNotFoundError, match="b.png"):
        cargar_conjunto_coco(carpeta, "lechuga")


def test_las_anotaciones_de_otra_clase_se_ignoran(raiz_sintetica):
    carpeta = _carpeta(raiz_sintetica)

    def cambiar(datos):
        datos["categories"].append({"id": 2, "name": "otra"})
        datos["annotations"].append(
            {"id": 99, "image_id": 2, "category_id": 2, "bbox": [0, 0, 5, 5], "iscrowd": 0}
        )

    _editar_json(carpeta, cambiar)
    assert len(cargar_conjunto_coco(carpeta, "lechuga")[1].cajas) == 1


def test_iscrowd_se_ignora_con_aviso(raiz_sintetica):
    carpeta = _carpeta(raiz_sintetica)
    _editar_json(carpeta, lambda d: d["annotations"][0].update(iscrowd=1))
    with pytest.warns(UserWarning, match="iscrowd"):
        imagenes = cargar_conjunto_coco(carpeta, "lechuga")
    assert len(imagenes[0].cajas) == 1


def test_sin_avisos_si_todo_esta_bien(raiz_sintetica):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cargar_conjunto_coco(_carpeta(raiz_sintetica), "lechuga")


def test_json_sin_las_secciones_esperadas(raiz_sintetica):
    carpeta = _carpeta(raiz_sintetica)
    (carpeta / NOMBRE_ANOTACIONES).write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="images"):
        cargar_conjunto_coco(carpeta, "lechuga")
