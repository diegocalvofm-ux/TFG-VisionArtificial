"""Pruebas de la lectura y escritura de imágenes (robustas con tildes y eñes en Windows)."""

import cv2
import numpy as np
import pytest

from src.pipeline.imagenes import EXTENSIONES, guardar_imagen, leer_imagen


def test_extensiones_admitidas():
    assert set(EXTENSIONES) == {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def test_leer_imagen_devuelve_bgr_de_tres_canales(crear_imagen, escribir_imagen, tmp_path):
    original = crear_imagen(30, 40)
    ruta = escribir_imagen(tmp_path / "foto.png", original)

    leida = leer_imagen(ruta)

    assert leida.shape == (30, 40, 3) and leida.dtype == np.uint8
    np.testing.assert_array_equal(leida, original)  # PNG no pierde información


def test_leer_imagen_en_gris_devuelve_tres_canales(escribir_imagen, tmp_path):
    gris = np.tile(np.arange(40, dtype=np.uint8), (30, 1))
    ruta = escribir_imagen(tmp_path / "gris.png", gris)

    assert leer_imagen(ruta).shape == (30, 40, 3)


def test_caso8_ida_y_vuelta_con_tildes_y_enes_en_la_ruta(crear_imagen, tmp_path):
    original = crear_imagen(30, 40)
    ruta = tmp_path / "carpeta ñandú" / "año_lechuga.png"
    ruta.parent.mkdir()

    guardar_imagen(ruta, original)

    assert ruta.is_file()
    np.testing.assert_array_equal(leer_imagen(ruta), original)


def test_guardar_imagen_jpg_crea_un_archivo_decodificable(crear_imagen, tmp_path):
    ruta = tmp_path / "año.jpg"

    guardar_imagen(ruta, crear_imagen(30, 40))

    decodificada = cv2.imdecode(np.fromfile(ruta, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decodificada.shape == (30, 40, 3)


def test_guardar_imagen_con_extension_desconocida_lanza_value_error(crear_imagen, tmp_path):
    with pytest.raises(ValueError):
        guardar_imagen(tmp_path / "foto.xyz", crear_imagen())


@pytest.mark.parametrize("contenido", [b"esto no es una imagen", b""], ids=["corrupto", "vacio"])
def test_leer_imagen_ilegible_lanza_value_error(tmp_path, contenido):
    ruta = tmp_path / "mala.jpg"
    ruta.write_bytes(contenido)

    with pytest.raises(ValueError):
        leer_imagen(ruta)


def test_leer_imagen_inexistente_lanza_os_error(tmp_path):
    with pytest.raises(OSError):
        leer_imagen(tmp_path / "no_existe.png")
