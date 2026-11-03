"""Pruebas de los adaptadores de los tres métodos, con modelos falsos (sin descargar nada)."""

import contextlib
import dataclasses
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.config import load_config
from src.eval.metodos import crear_fabricas, crear_linea_base, crear_owlvit, crear_rfdetr
from src.pipeline.detectors.owlvit import Cargado

RAIZ = Path(__file__).resolve().parents[2]


# --- OWL-ViT ---------------------------------------------------------------------------


class _Entradas(dict):
    def to(self, dispositivo):
        return self


class _ProcesadorFalso:
    """Imita OwlViTProcessor: registra la consulta y el umbral que recibe."""

    def __init__(self, cajas, puntuaciones):
        self.cajas, self.puntuaciones = cajas, puntuaciones
        self.consultas, self.umbrales = [], []

    def __call__(self, *, text, images, return_tensors):
        self.consultas.append(text)
        return _Entradas(pixel_values="x")

    def post_process_grounded_object_detection(
        self, outputs, threshold=0.1, target_sizes=None, text_labels=None
    ):
        self.umbrales.append(threshold)
        return [{"boxes": self.cajas, "scores": self.puntuaciones, "labels": None}]


def _cargador_owlvit(procesador, dispositivos=None):
    def cargador(config):
        if dispositivos is not None:
            dispositivos.append(config.dispositivo)
        return Cargado(procesador, lambda **entradas: "salidas", config.dispositivo,
                       contextlib.nullcontext)  # fmt: skip

    return cargador


def test_owlvit_usa_por_defecto_consulta_y_umbral_de_v0(config_real):
    v0 = load_config(RAIZ / "configs" / "v0.yaml")["owlvit"]
    procesador = _ProcesadorFalso(np.zeros((0, 4)), np.zeros(0))
    dispositivos = []
    metodo = crear_owlvit(config_real, RAIZ, _cargador_owlvit(procesador, dispositivos))
    metodo.predecir(np.zeros((50, 60, 3), dtype=np.uint8))

    assert procesador.consultas == [[[v0["consulta"]]]]  # misma consulta que la v0
    assert metodo.umbral == v0["umbral"]  # umbral operativo de la v0
    assert metodo.umbral_ap == 0.01
    assert procesador.umbrales == [0.01]  # el modelo se corre con el umbral bajo del AP
    assert dispositivos == ["cpu"]  # la evaluación fuerza CPU, aunque v0 diga auto
    assert metodo.tiene_confianza is True
    assert metodo.tiempo_carga_s >= 0


def test_owlvit_anota_los_parametros_usados_incluido_el_tope(config_real):
    v0 = load_config(RAIZ / "configs" / "v0.yaml")["owlvit"]
    procesador = _ProcesadorFalso(np.zeros((0, 4)), np.zeros(0))
    metodo = crear_owlvit(config_real, RAIZ, _cargador_owlvit(procesador))
    assert metodo.parametros["max_detecciones"] == 50  # el del YAML de evaluación, no el de v0
    assert v0["max_detecciones"] == 10
    assert metodo.parametros["consulta"] == v0["consulta"]
    assert metodo.parametros["modelo"] == v0["modelo"]
    assert metodo.parametros["nms_iou"] == v0["nms_iou"]


def test_owlvit_respeta_el_tope_de_detecciones(config_real):
    cajas = np.array([[i, i, i + 10, i + 10] for i in range(0, 200, 20)], dtype=float)
    procesador = _ProcesadorFalso(cajas, np.linspace(0.9, 0.5, len(cajas)))
    imagen = np.zeros((300, 300, 3), dtype=np.uint8)
    cfg = dataclasses.replace(config_real, owlvit_max_detecciones=3)
    metodo = crear_owlvit(cfg, RAIZ, _cargador_owlvit(procesador))
    detecciones = metodo.predecir(imagen)
    assert len(detecciones) == 3
    assert [d.score for d in detecciones] == sorted((d.score for d in detecciones), reverse=True)


def test_owlvit_con_valores_propios(config_real):
    cfg = dataclasses.replace(config_real, owlvit_consulta="una lechuga", owlvit_umbral=0.3)
    procesador = _ProcesadorFalso(np.zeros((0, 4)), np.zeros(0))
    metodo = crear_owlvit(cfg, RAIZ, _cargador_owlvit(procesador))
    metodo.predecir(np.zeros((50, 60, 3), dtype=np.uint8))
    assert procesador.consultas == [[["una lechuga"]]]
    assert metodo.umbral == 0.3


def test_owlvit_umbral_operativo_menor_que_el_de_ap(config_real):
    cfg = dataclasses.replace(config_real, owlvit_umbral=0.001)
    procesador = _ProcesadorFalso(np.zeros((0, 4)), np.zeros(0))
    with pytest.raises(ValueError, match="owlvit.umbral"):
        crear_owlvit(cfg, RAIZ, _cargador_owlvit(procesador))


def test_owlvit_config_base_inexistente(config_real, tmp_path):
    procesador = _ProcesadorFalso(np.zeros((0, 4)), np.zeros(0))
    with pytest.raises(FileNotFoundError, match="configuración"):
        crear_owlvit(config_real, tmp_path, _cargador_owlvit(procesador))


# --- RF-DETR ---------------------------------------------------------------------------


class _ModeloRFDETRFalso:
    """Imita ``RFDETR.predict``: devuelve un objeto con ``xyxy`` y ``confidence`` (supervision)."""

    def __init__(self, xyxy, confianza):
        self.xyxy = np.asarray(xyxy, dtype=float).reshape(-1, 4)
        self.confianza = np.asarray(confianza, dtype=float)
        self.llamadas = []

    def predict(self, imagen, threshold=0.5, **kwargs):
        self.llamadas.append({"imagen": imagen, "threshold": threshold, **kwargs})
        return SimpleNamespace(xyxy=self.xyxy, confidence=self.confianza)


def _cargador_rfdetr(modelo, registro=None):
    def cargador(ruta, dispositivo):
        if registro is not None:
            registro.append((ruta, dispositivo))
        return modelo

    return cargador


def test_rfdetr_convierte_bgr_a_rgb_y_usa_el_umbral_bajo(config_real):
    modelo = _ModeloRFDETRFalso([[0.4, 0.6, 10.2, 20.7]], [0.8])
    registro = []
    metodo = crear_rfdetr(config_real, RAIZ, _cargador_rfdetr(modelo, registro))
    imagen = np.zeros((30, 40, 3), dtype=np.uint8)
    imagen[:, :] = (10, 20, 30)  # BGR

    detecciones = metodo.predecir(imagen)

    recibida = modelo.llamadas[0]["imagen"]
    assert tuple(recibida[0, 0]) == (30, 20, 10)  # RGB: rfdetr lo espera así
    assert modelo.llamadas[0]["threshold"] == config_real.rfdetr_umbral_ap
    assert [(d.box, d.score) for d in detecciones] == [((0, 1, 10, 21), 0.8)]
    assert all(isinstance(v, int) for v in detecciones[0].box)
    assert registro == [(RAIZ / config_real.rfdetr_pesos, "cpu")]
    assert metodo.umbral == 0.5 and metodo.umbral_ap == 0.001
    assert metodo.tiene_confianza is True


def test_rfdetr_ordena_por_confianza_y_admite_cero_detecciones(config_real):
    modelo = _ModeloRFDETRFalso([[0, 0, 5, 5], [10, 10, 20, 20]], [0.2, 0.9])
    metodo = crear_rfdetr(config_real, RAIZ, _cargador_rfdetr(modelo))
    imagen = np.zeros((30, 40, 3), dtype=np.uint8)
    assert [d.score for d in metodo.predecir(imagen)] == [0.9, 0.2]

    vacio = _ModeloRFDETRFalso(np.zeros((0, 4)), np.zeros(0))
    metodo = crear_rfdetr(config_real, RAIZ, _cargador_rfdetr(vacio))
    assert metodo.predecir(imagen) == []


def test_rfdetr_pesos_inexistentes_sin_ruta_absoluta(config_real, tmp_path):
    with pytest.raises(FileNotFoundError) as error:
        crear_rfdetr(config_real, tmp_path)
    mensaje = str(error.value)
    assert config_real.rfdetr_pesos in mensaje
    assert str(tmp_path) not in mensaje  # el repositorio es público: ruta relativa


# --- Línea base -------------------------------------------------------------------------


def test_linea_base_no_tiene_confianza(config_real):
    metodo = crear_linea_base(config_real)
    assert metodo.tiene_confianza is False
    assert metodo.umbral is None and metodo.umbral_ap is None
    assert metodo.tiempo_carga_s is None
    assert metodo.parametros["area_min_frac"] == 0.005
    imagen = np.full((100, 100, 3), 128, dtype=np.uint8)
    assert metodo.predecir(imagen) == []


def test_crear_fabricas_tiene_los_tres_metodos(config_real):
    assert set(crear_fabricas(config_real, RAIZ)) == {"owlvit", "rfdetr", "linea_base_hsv"}
