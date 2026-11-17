"""Pruebas de src/train/etapa2/modelo.py: cabezas por grupo, timm (simulado), pesos y predecir.

Nada descarga pesos: el extractor es el diminuto del conftest o un ``timm`` falso.
"""

import sys
import types
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from src.data.schema import GROUPS  # noqa: E402
from src.experimentos.privacidad import contiene_ruta_absoluta, contiene_usuario  # noqa: E402
from src.train.etapa2 import modelo as modulo_modelo  # noqa: E402
from src.train.etapa2.aumentos import IMAGENET_DESV, IMAGENET_MEDIA  # noqa: E402
from src.train.etapa2.modelo import (  # noqa: E402
    ClasificadorEtapa2,
    cargar_modelo,
    crear_extractor_timm,
    guardar_pesos,
    predecir,
)


@pytest.fixture
def modelo(extractor_diminuto):
    torch.manual_seed(0)
    extractor, n = extractor_diminuto("efficientnet_b0", False)
    return ClasificadorEtapa2(extractor, n, dropout=0.2, resolucion=32,
                              arquitectura="efficientnet_b0")  # fmt: skip


def test_hay_una_salida_por_grupo_con_tantas_columnas_como_etiquetas(modelo):
    salidas = modelo(torch.zeros(2, 3, 32, 32))

    assert list(salidas) == list(GROUPS)
    assert [s.shape[1] for s in salidas.values()] == [4, 3, 3, 4, 3, 5]
    assert all(s.shape[0] == 2 for s in salidas.values())


def test_cada_cabeza_es_un_dropout_y_una_capa_lineal(modelo):
    for grupo, cabeza in modelo.cabezas.items():
        dropout, lineal = cabeza
        assert isinstance(dropout, nn.Dropout) and dropout.p == 0.2
        assert isinstance(lineal, nn.Linear)
        assert (lineal.in_features, lineal.out_features) == (4, len(GROUPS[grupo]))


def test_el_modelo_guarda_lo_necesario_para_reconstruirse(modelo):
    assert modelo.grupos == {g: tuple(e) for g, e in GROUPS.items()}
    assert (modelo.resolucion, modelo.dropout, modelo.arquitectura) == (32, 0.2, "efficientnet_b0")
    assert modelo.n_caracteristicas == 4


# --- timm (simulado) ----------------------------------------------------------------------


class _ModeloTimmFalso(nn.Module):
    def __init__(self, media=IMAGENET_MEDIA, desv=IMAGENET_DESV):
        super().__init__()
        self.num_features = 1280
        self.pretrained_cfg = {"mean": tuple(media), "std": tuple(desv)}


@pytest.fixture
def timm_falso(monkeypatch):
    """Un módulo ``timm`` de mentira que apunta las llamadas a ``create_model``."""
    modulo = types.ModuleType("timm")
    modulo.llamadas = []
    modulo.a_devolver = _ModeloTimmFalso()

    def create_model(nombre, **kwargs):
        modulo.llamadas.append((nombre, kwargs))
        return modulo.a_devolver

    modulo.create_model = create_model
    monkeypatch.setitem(sys.modules, "timm", modulo)
    return modulo


def test_crear_extractor_timm_pide_el_modelo_sin_su_clasificador(timm_falso):
    extractor, n = crear_extractor_timm("efficientnet_b0", True)

    assert timm_falso.llamadas == [
        ("efficientnet_b0", {"pretrained": True, "num_classes": 0, "drop_rate": 0.0})
    ]
    assert extractor is timm_falso.a_devolver and n == 1280


def test_crear_extractor_timm_sin_preentrenar_no_pide_pesos(timm_falso):
    crear_extractor_timm("efficientnet_b0", False)

    assert timm_falso.llamadas[0][1]["pretrained"] is False


def test_una_normalizacion_distinta_de_imagenet_da_error(timm_falso):
    timm_falso.a_devolver = _ModeloTimmFalso(media=(0.5, 0.5, 0.5), desv=(0.5, 0.5, 0.5))

    with pytest.raises(ValueError, match="normalización"):
        crear_extractor_timm("efficientnet_b0", True)


def test_sin_timm_el_error_dice_como_instalarlo(monkeypatch):
    monkeypatch.setitem(sys.modules, "timm", None)  # import timm -> ImportError

    with pytest.raises(ImportError, match="requirements-train.txt"):
        crear_extractor_timm("efficientnet_b0", True)


def test_el_modulo_no_importa_timm_al_cargarse():
    for nombre in ("modelo.py", "entrenar.py", "aumentos.py", "perdida.py"):
        codigo = (Path(modulo_modelo.__file__).parent / nombre).read_text(encoding="utf-8")
        for linea in codigo.splitlines():
            if linea.startswith(("import timm", "from timm")):
                pytest.fail(f"{nombre} importa timm al cargarse: {linea}")


# --- guardar y cargar ---------------------------------------------------------------------


def test_guardar_y_cargar_dan_las_mismas_salidas(modelo, tmp_path, extractor_diminuto):
    llamadas = []

    def fabrica(arquitectura, preentrenado):
        llamadas.append((arquitectura, preentrenado))
        return extractor_diminuto(arquitectura, preentrenado)

    modelo.eval()
    entrada = torch.rand(3, 3, 32, 32)
    ruta = tmp_path / "pesos" / "mejor.pt"
    guardar_pesos(ruta, modelo, {"epoca": 3, "criterio": 0.5})

    cargado = cargar_modelo(ruta, "cpu", crear_extractor=fabrica)

    assert llamadas == [("efficientnet_b0", False)]  # al cargar no se descargan pesos
    assert not cargado.training
    with torch.no_grad():
        antes, despues = modelo(entrada), cargado(entrada)
    for grupo in GROUPS:
        assert torch.allclose(antes[grupo], despues[grupo])


def test_los_pesos_llevan_sus_metadatos_y_ninguna_ruta(modelo, tmp_path):
    ruta = tmp_path / "ultimo.pt"
    guardar_pesos(ruta, modelo, {"epoca": 2, "criterio": None})

    datos = torch.load(ruta, map_location="cpu", weights_only=True)

    assert datos["arquitectura"] == "efficientnet_b0" and datos["resolucion"] == 32
    assert datos["grupos"] == {g: list(e) for g, e in GROUPS.items()}
    assert datos["media"] == list(IMAGENET_MEDIA) and datos["desv"] == list(IMAGENET_DESV)
    assert datos["epoca"] == 2 and datos["n_caracteristicas"] == 4
    texto = str({k: v for k, v in datos.items() if k != "estado"})
    assert not contiene_ruta_absoluta(texto) and not contiene_usuario(texto)


def test_unos_pesos_de_otra_forma_dan_error_al_cargar(modelo, tmp_path):
    ruta = tmp_path / "mejor.pt"
    guardar_pesos(ruta, modelo, {})

    def otro_extractor(arquitectura, preentrenado):
        return nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten()), 3

    with pytest.raises(ValueError, match="características"):
        cargar_modelo(ruta, "cpu", crear_extractor=otro_extractor)


# --- predecir -----------------------------------------------------------------------------


def _imagenes():
    generador = np.random.default_rng(0)
    return [generador.integers(0, 256, (20, 30, 3), dtype=np.uint8),
            generador.integers(0, 256, (40, 10, 3), dtype=np.uint8)]  # fmt: skip


def test_predecir_devuelve_una_etiqueta_valida_y_sus_probabilidades_por_grupo(modelo):
    resultados = predecir(modelo, _imagenes())

    assert len(resultados) == 2
    for resultado in resultados:
        assert list(resultado) == list(GROUPS)
        for grupo, salida in resultado.items():
            probabilidades = salida["probabilidades"]
            assert list(probabilidades) == list(GROUPS[grupo])
            assert sum(probabilidades.values()) == pytest.approx(1.0, abs=1e-5)
            assert salida["etiqueta"] == max(probabilidades, key=probabilidades.get)


def test_predecir_por_lotes_da_lo_mismo_y_en_el_mismo_orden(modelo):
    imagenes = _imagenes() * 3

    uno = predecir(modelo, imagenes, tamano_lote=1)
    todos = predecir(modelo, imagenes, tamano_lote=8)

    for a, b in zip(uno, todos, strict=True):
        for grupo in GROUPS:
            assert a[grupo]["etiqueta"] == b[grupo]["etiqueta"]
            for etiqueta, p in a[grupo]["probabilidades"].items():
                assert b[grupo]["probabilidades"][etiqueta] == pytest.approx(p, abs=1e-5)


def test_predecir_sin_imagenes_devuelve_una_lista_vacia(modelo):
    assert predecir(modelo, []) == []


def test_predecir_deja_el_modelo_en_el_modo_en_que_estaba(modelo):
    modelo.train()

    predecir(modelo, _imagenes())

    assert modelo.training


def test_predecir_nunca_devuelve_no_visible(modelo):
    for resultado in predecir(modelo, _imagenes()):
        assert all(salida["etiqueta"] != "NO_VISIBLE" for salida in resultado.values())
