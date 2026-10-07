"""Pruebas del detector OWL-ViT con un procesador y un modelo falsos (sin descargar nada).

El procesador falso imita la interfaz de OwlViTProcessor de transformers 5.x:
``procesador(text=..., images=..., return_tensors=...)`` y
``post_process_grounded_object_detection(outputs, threshold, target_sizes)``, que devuelve una
lista con un diccionario ``scores`` / ``labels`` / ``boxes`` por imagen.
"""

import copy
import dataclasses
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pytest

from src.config import load_config
from src.pipeline.detectors import DETECTORS, crear_detector, nombres_detectores
from src.pipeline.detectors.owlvit import (
    Cargado,
    OwlViTConfig,
    OwlViTDetector,
    crear_owlvit,
    resolver_dispositivo,
)
from src.pipeline.run import crear_parser, main

RAIZ = Path(__file__).resolve().parents[2]
V0_YAML = RAIZ / "configs" / "v0.yaml"
MENSAJE_INSTALAR = "pip install -r requirements-detector.txt"


# --- Dobles de prueba -------------------------------------------------------------------


class EntradasFalsas(dict):
    """Imita el BatchFeature del procesador: un diccionario con ``.to(dispositivo)``."""

    def __init__(self):
        super().__init__(pixel_values="pixel_values_falsos")
        self.dispositivo = None

    def to(self, dispositivo):
        self.dispositivo = dispositivo
        return self


class ProcesadorFalso:
    def __init__(self, resultado):
        self.resultado = resultado
        self.llamadas = []
        self.llamadas_post = []
        self.entradas = []

    def __call__(self, *, text, images, return_tensors):
        self.llamadas.append({"text": text, "images": images, "return_tensors": return_tensors})
        self.entradas.append(EntradasFalsas())
        return self.entradas[-1]

    def post_process_grounded_object_detection(
        self, outputs, threshold=0.1, target_sizes=None, text_labels=None
    ):
        self.llamadas_post.append(
            {"outputs": outputs, "threshold": threshold, "target_sizes": target_sizes}
        )
        return [self.resultado]


class ModeloFalso:
    def __init__(self):
        self.llamadas = []

    def __call__(self, **entradas):
        self.llamadas.append(sorted(entradas))
        return "salidas_del_modelo"


class ContextoFalso:
    """Imita ``torch.no_grad``: se llama para obtener el contexto y cuenta las entradas."""

    def __init__(self):
        self.entradas = 0

    def __call__(self):
        return self

    def __enter__(self):
        self.entradas += 1

    def __exit__(self, *args):
        return False


class Montaje:
    """Detector con procesador y modelo falsos, y registro de lo que se les pidió."""

    def __init__(self, config, resultado, espera=0.0):
        self.procesador = ProcesadorFalso(resultado)
        self.modelo = ModeloFalso()
        self.contexto = ContextoFalso()
        self.cargas = 0

        def cargador(cfg):
            self.cargas += 1
            time.sleep(espera)
            return Cargado(self.procesador, self.modelo, "cpu", self.contexto)

        self.cargador = cargador
        self.detector = OwlViTDetector(config, cargador=cargador)


def resultado(cajas, puntuaciones):
    """Lo que devolvería el postprocesado de la librería para una imagen."""
    return {
        "scores": np.array(puntuaciones, dtype=float),
        "boxes": np.array(cajas, dtype=float).reshape(-1, 4),
        "labels": np.zeros(len(puntuaciones), dtype=int),
    }


def cajas_separadas(n, lado=8):
    """n cajas que no se solapan, en una fila: (i*10, 0, i*10 + lado, lado)."""
    return [(i * 10, 0, i * 10 + lado, lado) for i in range(n)]


@pytest.fixture
def config():
    return OwlViTConfig(
        modelo="google/owlvit-base-patch32",
        consulta="a photo of a lettuce",
        umbral=0.1,
        max_detecciones=10,
        nms_iou=0.5,
        dispositivo="cpu",
    )


def detectar(config, cajas, puntuaciones, imagen, **cambios):
    montaje = Montaje(dataclasses.replace(config, **cambios), resultado(cajas, puntuaciones))
    return montaje.detector.detect(imagen)


# --- Casos 1 a 7: postprocesado ---------------------------------------------------------


def test_caso1_las_cajas_por_debajo_del_umbral_se_descartan(config, crear_imagen):
    cajas = cajas_separadas(4)
    puntuaciones = [0.9, 0.05, 0.1, 0.5]  # la tercera vale justo el umbral: no está por encima

    montaje = Montaje(config, resultado(cajas, puntuaciones))
    detecciones = montaje.detector.detect(crear_imagen(48, 64))

    assert [d.box for d in detecciones] == [cajas[0], cajas[3]]
    assert [d.score for d in detecciones] == [0.9, 0.5]
    assert montaje.procesador.llamadas_post[0]["threshold"] == 0.1  # y se lo pide a la librería


def test_caso2_las_cajas_se_expresan_en_pixeles_de_la_imagen_original(config, crear_imagen):
    montaje = Montaje(config, resultado(cajas_separadas(1), [0.9]))

    montaje.detector.detect(crear_imagen(alto=48, ancho=64))

    llamada = montaje.procesador.llamadas_post[0]
    assert llamada["target_sizes"] == [(48, 64)]  # (alto, ancho) de la imagen original
    assert llamada["outputs"] == "salidas_del_modelo"


def test_caso2_las_cajas_se_recortan_a_la_imagen_y_se_redondean(config, crear_imagen):
    imagen = crear_imagen(alto=48, ancho=64)
    cajas = [(-10.4, -5.0, 30.6, 70.0), (50.2, 10.4, 100.0, 29.6)]

    detecciones = detectar(config, cajas, [0.9, 0.8], imagen)

    assert [d.box for d in detecciones] == [(0, 0, 31, 48), (50, 10, 64, 30)]
    for deteccion in detecciones:
        assert all(type(valor) is int for valor in deteccion.box)


@pytest.mark.parametrize(
    "caja",
    [
        (10, 10, 10, 30),  # ancho 0
        (10, 10, 30, 10),  # alto 0
        (30, 30, 10, 10),  # invertida
        (100, 100, 120, 120),  # fuera de la imagen: al recortar se queda sin área
        (10, 10, 10.3, 30),  # tan fina que al redondear queda con ancho 0
        (np.nan, 10, 30, 30),  # no finita
    ],
    ids=["ancho_0", "alto_0", "invertida", "fuera", "fina", "nan"],
)
def test_caso3_una_caja_degenerada_se_descarta(config, crear_imagen, caja):
    valida = (40, 5, 60, 25)

    detecciones = detectar(config, [caja, valida], [0.9, 0.8], crear_imagen(48, 64))

    assert [d.box for d in detecciones] == [valida]


def test_una_puntuacion_no_finita_se_descarta(config, crear_imagen):
    cajas = cajas_separadas(2)

    detecciones = detectar(config, cajas, [np.nan, 0.8], crear_imagen(48, 64))

    assert [d.box for d in detecciones] == [cajas[1]]


def test_caso4_el_nms_conserva_la_de_mayor_puntuacion_entre_dos_muy_solapadas(
    config, crear_imagen
):
    cajas = [(0, 0, 20, 20), (1, 1, 21, 21)]  # IoU = 0.82

    detecciones = detectar(config, cajas, [0.6, 0.9], crear_imagen(48, 64))

    assert [(d.box, d.score) for d in detecciones] == [((1, 1, 21, 21), 0.9)]


def test_caso5_el_nms_conserva_dos_cajas_que_no_se_solapan(config, crear_imagen):
    cajas = [(0, 0, 20, 20), (30, 0, 50, 20)]

    detecciones = detectar(config, cajas, [0.6, 0.9], crear_imagen(48, 64))

    assert [d.box for d in detecciones] == [cajas[1], cajas[0]]


def test_el_umbral_de_iou_del_nms_sale_de_la_configuracion(config, crear_imagen):
    cajas = [(0, 0, 20, 20), (10, 0, 30, 20)]  # IoU = 0.33

    estricto = detectar(config, cajas, [0.9, 0.8], crear_imagen(48, 64), nms_iou=0.2)
    permisivo = detectar(config, cajas, [0.9, 0.8], crear_imagen(48, 64), nms_iou=0.5)

    assert len(estricto) == 1 and len(permisivo) == 2


def test_caso6_el_resultado_va_de_mayor_a_menor_puntuacion_y_respeta_el_maximo(
    config, crear_imagen
):
    cajas = cajas_separadas(6)
    puntuaciones = [0.3, 0.9, 0.5, 0.7, 0.2, 0.6]

    detecciones = detectar(config, cajas, puntuaciones, crear_imagen(48, 64), max_detecciones=3)

    assert [d.score for d in detecciones] == [0.9, 0.7, 0.6]
    assert [d.box for d in detecciones] == [cajas[1], cajas[3], cajas[5]]


def test_el_maximo_se_aplica_despues_del_nms(config, crear_imagen):
    # La segunda caja es casi igual a la primera y el NMS la elimina. Si el tope de 2 se
    # aplicara antes, la segunda ocuparía un hueco y la tercera se perdería.
    cajas = [(0, 0, 20, 20), (1, 1, 21, 21), (30, 0, 50, 20), (30, 25, 50, 45)]

    detecciones = detectar(
        config, cajas, [0.95, 0.94, 0.9, 0.8], crear_imagen(48, 64), max_detecciones=2
    )

    assert [d.box for d in detecciones] == [cajas[0], cajas[2]]


@pytest.mark.parametrize("vacio", [resultado([], []), {"scores": [], "boxes": [], "labels": []}])
def test_caso7_sin_detecciones_da_lista_vacia_sin_error(config, crear_imagen, vacio):
    montaje = Montaje(config, vacio)

    assert montaje.detector.detect(crear_imagen(48, 64)) == []


def test_si_todas_quedan_por_debajo_del_umbral_da_lista_vacia(config, crear_imagen):
    assert detectar(config, cajas_separadas(2), [0.01, 0.02], crear_imagen(48, 64)) == []


def test_las_detecciones_son_del_tipo_del_pipeline(config, crear_imagen):
    detecciones = detectar(config, cajas_separadas(1), [0.9], crear_imagen(48, 64))

    assert type(detecciones[0].score) is float
    assert detecciones[0].label == "lettuce"


def test_admite_tensores_de_torch_como_los_que_devuelve_la_libreria(config, crear_imagen):
    torch = pytest.importorskip("torch")
    cajas = torch.tensor([[0.0, 0.0, 20.0, 20.0], [30.0, 0.0, 50.0, 20.0]])
    montaje = Montaje(config, {"scores": torch.tensor([0.5, 0.75]), "boxes": cajas, "labels": None})

    detecciones = montaje.detector.detect(crear_imagen(48, 64))

    assert [d.box for d in detecciones] == [(30, 0, 50, 20), (0, 0, 20, 20)]
    assert [d.score for d in detecciones] == pytest.approx([0.75, 0.5])


# --- Cómo se llama a procesador y modelo ------------------------------------------------


def test_se_pasa_la_consulta_y_la_imagen_en_rgb_al_procesador(config, crear_imagen):
    imagen = crear_imagen(48, 64)
    imagen[0, 0] = (10, 20, 30)  # BGR
    montaje = Montaje(config, resultado([], []))

    montaje.detector.detect(imagen)

    llamada = montaje.procesador.llamadas[0]
    assert llamada["text"] == [["a photo of a lettuce"]]
    assert llamada["return_tensors"] == "pt"
    assert llamada["images"].shape == (48, 64, 3)
    assert tuple(llamada["images"][0, 0]) == (30, 20, 10)  # RGB: el canal rojo primero
    assert imagen[0, 0].tolist() == [10, 20, 30]  # y la imagen original no se toca


def test_las_entradas_van_al_dispositivo_y_el_modelo_corre_sin_gradientes(config, crear_imagen):
    montaje = Montaje(config, resultado([], []))

    montaje.detector.detect(crear_imagen(48, 64))
    montaje.detector.detect(crear_imagen(48, 64))

    assert [e.dispositivo for e in montaje.procesador.entradas] == ["cpu", "cpu"]
    assert montaje.modelo.llamadas == [["pixel_values"], ["pixel_values"]]
    assert montaje.contexto.entradas == 2


# --- Carga única y tiempo de carga ------------------------------------------------------


def test_el_modelo_se_carga_una_sola_vez_al_crear_el_detector(config, crear_imagen):
    montaje = Montaje(config, resultado([], []))
    assert montaje.cargas == 1  # ya cargado antes de ver ninguna imagen

    for _ in range(3):
        montaje.detector.detect(crear_imagen(48, 64))

    assert montaje.cargas == 1
    assert len(montaje.modelo.llamadas) == 3


def test_el_tiempo_de_carga_se_mide_aparte_de_la_inferencia(config, crear_imagen):
    montaje = Montaje(config, resultado([], []), espera=0.05)
    tiempo = montaje.detector.tiempo_carga_s

    montaje.detector.detect(crear_imagen(48, 64))

    assert isinstance(tiempo, float) and tiempo >= 0.04  # incluye la espera de la carga
    assert montaje.detector.tiempo_carga_s == tiempo  # la inferencia no lo modifica


def test_el_cargador_recibe_la_configuracion(config):
    recibido = []

    def cargador(cfg):
        recibido.append(cfg)
        return Cargado(ProcesadorFalso(resultado([], [])), ModeloFalso(), "cpu", ContextoFalso())

    OwlViTDetector(config, cargador=cargador)

    assert recibido == [config]


# --- Caso 8: transformers es opcional ---------------------------------------------------


def test_caso8_sin_transformers_el_error_es_claro(monkeypatch, config):
    monkeypatch.setitem(sys.modules, "transformers", None)  # como si no estuviera instalado

    with pytest.raises(ImportError, match=MENSAJE_INSTALAR) as error:
        OwlViTDetector(config)

    assert "transformers" in str(error.value)


def test_caso8_sin_transformers_el_error_tambien_sale_por_el_registro(monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", None)

    with pytest.raises(ImportError, match=MENSAJE_INSTALAR):
        crear_detector("owlvit", load_config(V0_YAML))


def test_importar_el_registro_no_importa_transformers_ni_torch():
    # En un proceso aparte: en este, otras pruebas ya pueden haber importado torch.
    codigo = (
        "import sys, src.pipeline.run, src.pipeline.detectors.owlvit\n"
        "cargados = [m for m in ('transformers', 'torch') if m in sys.modules]\n"
        "sys.exit('importados al arrancar: ' + ', '.join(cargados) if cargados else 0)\n"
    )

    proceso = subprocess.run(
        [sys.executable, "-c", codigo], cwd=RAIZ, capture_output=True, text=True, encoding="utf-8"
    )

    assert proceso.returncode == 0, proceso.stderr


# --- Caso 9: registro -------------------------------------------------------------------


def test_caso9_owlvit_esta_en_el_registro_y_el_error_de_desconocido_lo_lista():
    assert "owlvit" in DETECTORS
    assert "owlvit" in nombres_detectores()

    with pytest.raises(ValueError) as error:
        crear_detector("nope", {})

    assert "owlvit" in str(error.value) and "whole" in str(error.value)


# --- Dispositivo ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("nombre", "hay_cuda", "esperado"),
    [
        ("auto", True, "cuda"),
        ("auto", False, "cpu"),
        ("cpu", True, "cpu"),
        ("cpu", False, "cpu"),
        ("cuda", True, "cuda"),
    ],
)
def test_resolver_dispositivo(nombre, hay_cuda, esperado):
    assert resolver_dispositivo(nombre, hay_cuda) == esperado


def test_pedir_cuda_sin_gpu_es_un_error_y_no_cae_a_cpu_en_silencio():
    with pytest.raises(ValueError, match="cuda"):
        resolver_dispositivo("cuda", False)


def test_dispositivo_desconocido_es_un_error():
    with pytest.raises(ValueError, match="tpu"):
        resolver_dispositivo("tpu", True)


# --- Configuración ----------------------------------------------------------------------

CFG_VALIDA = {
    "owlvit": {
        "modelo": "google/owlvit-base-patch32",
        "consulta": "a photo of a lettuce",
        "umbral": 0.1,
        "max_detecciones": 10,
        "nms_iou": 0.5,
        "dispositivo": "auto",
    }
}


def _cfg_con(**cambios):
    cfg = copy.deepcopy(CFG_VALIDA)
    cfg["owlvit"].update(cambios)
    return cfg


def test_la_configuracion_valida_se_lee_entera():
    leida = OwlViTConfig.desde_dict(CFG_VALIDA)

    assert leida == OwlViTConfig(**CFG_VALIDA["owlvit"])


def test_un_entero_vale_como_umbral_y_se_guarda_como_float():
    leida = OwlViTConfig.desde_dict(_cfg_con(umbral=1, nms_iou=0))

    assert leida.umbral == 1.0 and type(leida.umbral) is float
    assert leida.nms_iou == 0.0 and type(leida.nms_iou) is float


def test_sin_bloque_owlvit_es_un_error():
    with pytest.raises(ValueError, match="owlvit"):
        OwlViTConfig.desde_dict({"detector": "owlvit"})


@pytest.mark.parametrize("clave", list(CFG_VALIDA["owlvit"]))
def test_si_falta_una_clave_es_un_error_que_la_nombra(clave):
    cfg = copy.deepcopy(CFG_VALIDA)
    del cfg["owlvit"][clave]

    with pytest.raises(ValueError, match=clave):
        OwlViTConfig.desde_dict(cfg)


@pytest.mark.parametrize(
    ("clave", "valor"),
    [
        ("umbral", 1.5),
        ("umbral", -0.1),
        ("umbral", "0.1"),
        ("umbral", True),
        ("nms_iou", 2),
        ("nms_iou", -1),
        ("max_detecciones", 0),
        ("max_detecciones", 2.5),
        ("max_detecciones", True),
        ("dispositivo", "tpu"),
        ("consulta", ""),
        ("consulta", "   "),
        ("consulta", 5),
        ("modelo", ""),
    ],
)
def test_un_valor_invalido_es_un_error_que_nombra_la_clave(clave, valor):
    with pytest.raises(ValueError, match=clave):
        OwlViTConfig.desde_dict(_cfg_con(**{clave: valor}))


def test_el_v0_yaml_real_tiene_la_configuracion_de_owlvit_completa():
    leida = OwlViTConfig.desde_dict(load_config(V0_YAML))

    assert leida.modelo == "google/owlvit-base-patch32"
    assert leida.dispositivo in {"auto", "cpu", "cuda"}


# --- Línea de comandos: --umbral y --consulta -------------------------------------------


class DetectorConCarga:
    """Detector falso con tiempo de carga, para comprobar que el pipeline lo muestra."""

    tiempo_carga_s = 0.25

    def detect(self, imagen):
        return []


@pytest.fixture
def entrada(tmp_path, crear_imagen, escribir_imagen):
    carpeta = tmp_path / "entrada"
    escribir_imagen(carpeta / "a.png", crear_imagen())
    return carpeta


@pytest.fixture
def fabrica_espia(monkeypatch):
    """Registra un "owlvit" falso que guarda la configuración con la que se crea."""
    recibidas = []

    def fabrica(cfg):
        recibidas.append(copy.deepcopy(cfg))
        return DetectorConCarga()

    monkeypatch.setitem(DETECTORS, "owlvit", fabrica)
    return recibidas


def test_las_opciones_umbral_y_consulta_no_tienen_valor_por_defecto():
    argumentos = crear_parser().parse_args(["--input", "fotos"])

    assert argumentos.umbral is None and argumentos.consulta is None


def test_umbral_y_consulta_llegan_al_detector_sin_tocar_el_yaml(
    tmp_path, entrada, fabrica_espia
):
    antes = V0_YAML.read_bytes()

    codigo = main(
        ["--input", str(entrada), "--out", str(tmp_path / "s"), "--detector", "owlvit"]
        + ["--umbral", "0.3", "--consulta", "a head of lettuce"]
    )

    assert codigo == 0
    bloque = fabrica_espia[0]["owlvit"]
    assert bloque["umbral"] == 0.3 and bloque["consulta"] == "a head of lettuce"
    assert bloque["max_detecciones"] == load_config(V0_YAML)["owlvit"]["max_detecciones"]
    assert V0_YAML.read_bytes() == antes


def test_sin_las_opciones_se_usan_los_valores_del_yaml(tmp_path, entrada, fabrica_espia):
    codigo = main(["--input", str(entrada), "--out", str(tmp_path / "s"), "--detector", "owlvit"])

    assert codigo == 0
    assert fabrica_espia[0]["owlvit"] == load_config(V0_YAML)["owlvit"]


def test_con_otro_detector_las_opciones_se_ignoran_con_aviso(tmp_path, entrada):
    argv = ["--input", str(entrada), "--out", str(tmp_path / "s"), "--detector", "whole"]

    with pytest.warns(UserWarning, match="umbral"):
        codigo = main([*argv, "--umbral", "0.3"])

    assert codigo == 0


def test_sin_las_opciones_no_hay_avisos(tmp_path, entrada):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        codigo = main(["--input", str(entrada), "--out", str(tmp_path / "s")])

    assert codigo == 0


def test_main_muestra_el_tiempo_de_carga_del_modelo(tmp_path, entrada, fabrica_espia, capsys):
    main(["--input", str(entrada), "--out", str(tmp_path / "s"), "--detector", "owlvit"])

    assert "Carga del modelo: 0.25 s" in capsys.readouterr().out


def test_un_umbral_invalido_por_linea_de_comandos_devuelve_2(
    tmp_path, entrada, monkeypatch, capsys
):
    def cargador(cfg):
        return Cargado(ProcesadorFalso(resultado([], [])), ModeloFalso(), "cpu", ContextoFalso())

    monkeypatch.setitem(DETECTORS, "owlvit", lambda cfg: crear_owlvit(cfg, cargador=cargador))

    codigo = main(
        ["--input", str(entrada), "--out", str(tmp_path / "s"), "--detector", "owlvit"]
        + ["--umbral", "2"]
    )

    assert codigo == 2
    assert "umbral" in capsys.readouterr().err


def test_main_sin_transformers_devuelve_2_con_el_mensaje_de_instalacion(
    tmp_path, entrada, monkeypatch, capsys
):
    monkeypatch.setitem(sys.modules, "transformers", None)

    codigo = main(["--input", str(entrada), "--out", str(tmp_path / "s"), "--detector", "owlvit"])

    assert codigo == 2
    assert MENSAJE_INSTALAR in capsys.readouterr().err


def test_main_con_un_fallo_de_descarga_del_modelo_devuelve_2(
    tmp_path, entrada, monkeypatch, capsys
):
    def sin_red(cfg):
        raise OSError("no hay conexión con huggingface.co")

    monkeypatch.setitem(DETECTORS, "owlvit", sin_red)

    codigo = main(["--input", str(entrada), "--out", str(tmp_path / "s"), "--detector", "owlvit"])

    assert codigo == 2
    assert "no hay conexión" in capsys.readouterr().err


# --- Caso 10: modelo real (lento, se salta por defecto) ---------------------------------


@pytest.mark.slow
def test_caso10_con_el_modelo_real_se_ejecuta_y_las_cajas_son_validas(crear_imagen):
    pytest.importorskip("transformers")
    pytest.importorskip("torch")
    cfg = load_config(V0_YAML)
    umbral, maximo = cfg["owlvit"]["umbral"], cfg["owlvit"]["max_detecciones"]
    alto, ancho = 480, 640

    detector = crear_detector("owlvit", cfg)  # la primera vez descarga el modelo
    detecciones = detector.detect(crear_imagen(alto, ancho))

    # Solo se comprueba que corre y que lo que devuelve es coherente, no que halle una lechuga.
    assert detector.tiempo_carga_s >= 0
    assert len(detecciones) <= maximo
    for d in detecciones:
        x1, y1, x2, y2 = d.box
        assert 0 <= x1 < x2 <= ancho and 0 <= y1 < y2 <= alto
        assert umbral < d.score <= 1.0
    puntuaciones = [d.score for d in detecciones]
    assert puntuaciones == sorted(puntuaciones, reverse=True)
