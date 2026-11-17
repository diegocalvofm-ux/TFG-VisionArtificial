"""Pruebas de src/train/etapa2/aumentos.py: transformaciones de train y eval y el Dataset.

La prueba central: los aumentos nunca cambian el tono (falsearía el grupo ``color``). Como mucho
brillo y contraste suaves, que escalan o desplazan los tres canales por igual.
"""

import colorsys
from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
from torchvision.transforms import v2  # noqa: E402

from src.pipeline.imagenes import guardar_imagen  # noqa: E402
from src.train.etapa2.aumentos import (  # noqa: E402
    COLOR_RELLENO,
    IMAGENET_DESV,
    IMAGENET_MEDIA,
    Cuadrar,
    DatasetRecortes,
    Giro90Aleatorio,
    a_tensor_rgb,
    cuadrar,
    transformacion_eval,
    transformacion_train,
)
from src.train.etapa2.config import Aumentos  # noqa: E402

AUMENTOS = Aumentos(
    escala_recorte=(0.9, 1.0),
    proporcion_recorte=(1.0, 1.0),
    volteo_horizontal=True,
    volteo_vertical=True,
    giros_90=True,
    rotacion_grados=10,
    brillo=0.2,
    contraste=0.2,
)
NADA = replace(AUMENTOS, volteo_horizontal=False, volteo_vertical=False, giros_90=False,
               rotacion_grados=0, brillo=0.0, contraste=0.0)  # fmt: skip

# Únicas transformaciones que puede tener el pipeline de train.
PERMITIDAS = (
    Cuadrar, v2.RandomResizedCrop, v2.RandomHorizontalFlip, v2.RandomVerticalFlip,
    Giro90Aleatorio, v2.RandomRotation, v2.ColorJitter, v2.ToDtype, v2.Normalize,
)  # fmt: skip


def _tipos(transformacion) -> list[type]:
    return [type(t) for t in transformacion.transforms]


def _desnormalizar(tensor):
    """De la salida normalizada a valores RGB en [0, 1]."""
    media = torch.tensor(IMAGENET_MEDIA).view(3, 1, 1)
    desv = torch.tensor(IMAGENET_DESV).view(3, 1, 1)
    return tensor * desv + media


def test_el_color_de_relleno_es_la_media_de_imagenet():
    assert COLOR_RELLENO == (124, 116, 104)


def test_a_tensor_rgb_pasa_de_bgr_a_rgb_en_canales_primero():
    bgr = np.zeros((2, 3, 3), dtype=np.uint8)
    bgr[..., 0] = 255  # azul en BGR

    tensor = a_tensor_rgb(bgr)

    assert tensor.shape == (3, 2, 3) and tensor.dtype == torch.uint8
    assert tensor[2].eq(255).all() and tensor[0].eq(0).all()


def test_cuadrar_conserva_la_proporcion_y_rellena_con_el_color_medio():
    imagen = torch.full((3, 20, 40), 200, dtype=torch.uint8)  # el doble de ancha que alta

    salida = cuadrar(imagen, 32)

    assert salida.shape == (3, 32, 32)
    # 20x40 -> 16x32, centrada: filas 8 a 23 con la imagen y el resto con el relleno
    assert salida[:, 8:24, :].eq(200).all()
    relleno = torch.tensor(COLOR_RELLENO, dtype=torch.uint8).view(3, 1, 1)
    assert salida[:, :8, :].eq(relleno).all() and salida[:, 24:, :].eq(relleno).all()


def test_cuadrar_una_imagen_alta_rellena_a_los_lados():
    salida = cuadrar(torch.full((3, 40, 10), 50, dtype=torch.uint8), 32)

    assert salida.shape == (3, 32, 32)
    assert salida[:, :, 12:20].eq(50).all()  # 40x10 -> 32x8, centrada en las columnas 12-19


def test_eval_da_un_cuadrado_normalizado_con_imagenet():
    color = (200, 100, 50)
    imagen = torch.tensor(color, dtype=torch.uint8).view(3, 1, 1).expand(3, 30, 50).clone()

    salida = transformacion_eval(32)(imagen)

    assert salida.shape == (3, 32, 32) and salida.dtype == torch.float32
    esperado = [(c / 255 - m) / d for c, m, d in zip(color, IMAGENET_MEDIA, IMAGENET_DESV,
                                                    strict=True)]  # fmt: skip
    centro = salida[:, 16, 16]
    assert torch.allclose(centro, torch.tensor(esperado), atol=1e-4)
    # el relleno, el color medio de ImageNet, queda en ~0 tras normalizar
    assert salida[:, 0, 16].abs().max() < 0.01


def test_eval_no_tiene_aumentos():
    assert _tipos(transformacion_eval(32)) == [Cuadrar, v2.ToDtype, v2.Normalize]


def test_el_pipeline_de_train_solo_usa_transformaciones_permitidas():
    tipos = _tipos(transformacion_train(32, AUMENTOS))

    assert all(t in PERMITIDAS for t in tipos), tipos
    assert tipos[0] is Cuadrar and tipos[-2:] == [v2.ToDtype, v2.Normalize]


def test_el_cambio_de_color_nunca_toca_la_saturacion_ni_el_tono():
    (jitter,) = [t for t in transformacion_train(32, AUMENTOS).transforms
                 if isinstance(t, v2.ColorJitter)]  # fmt: skip

    assert jitter.saturation is None and jitter.hue is None
    assert tuple(jitter.brightness) == pytest.approx((0.8, 1.2))
    assert tuple(jitter.contrast) == pytest.approx((0.8, 1.2))


def test_el_recorte_aleatorio_usa_la_escala_y_la_proporcion_del_yaml():
    (recorte,) = [t for t in transformacion_train(32, AUMENTOS).transforms
                  if isinstance(t, v2.RandomResizedCrop)]  # fmt: skip

    assert tuple(recorte.scale) == (0.9, 1.0) and tuple(recorte.ratio) == (1.0, 1.0)
    assert tuple(recorte.size) == (32, 32)


def test_con_brillo_y_contraste_a_cero_no_hay_ningun_cambio_de_color():
    tipos = _tipos(transformacion_train(32, replace(AUMENTOS, brillo=0.0, contraste=0.0)))

    assert v2.ColorJitter not in tipos


def test_cada_aumento_se_puede_desactivar():
    assert _tipos(transformacion_train(32, NADA)) == [
        Cuadrar, v2.RandomResizedCrop, v2.ToDtype, v2.Normalize
    ]  # fmt: skip


def test_el_brillo_y_el_contraste_al_maximo_no_cambian_el_tono():
    color = (180, 120, 60)  # naranja medio: sin saturar ningún canal al aclarar u oscurecer
    imagen = torch.tensor(color, dtype=torch.uint8).view(3, 1, 1).expand(3, 32, 32).clone()
    solo_color = replace(NADA, escala_recorte=(1.0, 1.0), brillo=0.2, contraste=0.2)
    transformacion = transformacion_train(32, solo_color)
    tono = colorsys.rgb_to_hsv(*(c / 255 for c in color))[0]

    torch.manual_seed(0)
    for _ in range(25):
        rgb = _desnormalizar(transformacion(imagen))
        r, g, b = (float(v) for v in rgb[:, 16, 16])
        assert colorsys.rgb_to_hsv(r, g, b)[0] == pytest.approx(tono, abs=0.01)


def test_los_aumentos_geometricos_no_cambian_los_colores_de_la_imagen():
    # Sin rotación ni recorte que interpolen: volteos y giros solo cambian píxeles de sitio.
    imagen = torch.randint(0, 256, (3, 32, 32), dtype=torch.uint8)
    geometricos = replace(NADA, escala_recorte=(1.0, 1.0), volteo_horizontal=True,
                          volteo_vertical=True, giros_90=True)  # fmt: skip
    transformacion = transformacion_train(32, geometricos)
    original = sorted(map(tuple, imagen.reshape(3, -1).T.tolist()))

    torch.manual_seed(1)
    for _ in range(10):
        rgb = (_desnormalizar(transformacion(imagen)) * 255).round().to(torch.uint8)
        assert sorted(map(tuple, rgb.reshape(3, -1).T.tolist())) == original


def test_giro90_aleatorio_gira_siempre_un_multiplo_de_90_grados():
    imagen = torch.arange(3 * 4 * 4, dtype=torch.uint8).reshape(3, 4, 4)
    giros = [torch.rot90(imagen, k, dims=(-2, -1)) for k in range(4)]
    vistos = set()

    torch.manual_seed(2)
    for _ in range(40):
        salida = Giro90Aleatorio()(imagen)
        (k,) = [k for k, g in enumerate(giros) if torch.equal(salida, g)]
        vistos.add(k)

    assert vistos == {0, 1, 2, 3}


def test_el_dataset_devuelve_imagen_objetivos_e_indice(tmp_path):
    rutas = []
    for i in range(2):
        ruta = tmp_path / f"r{i}.jpg"
        guardar_imagen(ruta, np.full((16, 24, 3), 60 * i, dtype=np.uint8))
        rutas.append(ruta)
    objetivos = np.array([[0, -1, 1, 0, 2, 0], [1, 2, -1, 3, 0, 4]], dtype=np.int64)

    dataset = DatasetRecortes(rutas, objetivos, transformacion_eval(32))
    imagen, objetivo, indice = dataset[1]

    assert len(dataset) == 2 and indice == 1
    assert imagen.shape == (3, 32, 32) and imagen.dtype == torch.float32
    assert objetivo.dtype == torch.int64 and objetivo.tolist() == [1, 2, -1, 3, 0, 4]


def test_el_dataset_lee_rutas_con_tildes_y_enes(tmp_path):
    ruta = tmp_path / "daño_ñ" / "recorte_á.jpg"
    ruta.parent.mkdir()
    guardar_imagen(ruta, np.full((10, 10, 3), 90, dtype=np.uint8))

    imagen, _, _ = DatasetRecortes([ruta], np.zeros((1, 6), dtype=np.int64),
                                   transformacion_eval(32))[0]  # fmt: skip

    assert imagen.shape == (3, 32, 32)


def test_el_dataset_rechaza_rutas_y_objetivos_de_distinta_longitud(tmp_path):
    with pytest.raises(ValueError, match="objetivos"):
        DatasetRecortes([tmp_path / "a.jpg"], np.zeros((2, 6), dtype=np.int64),
                        transformacion_eval(32))  # fmt: skip
