"""Medición en el pipeline: env ampliado, medianas con calentamiento, memoria y carga del modelo.

Los tiempos se simulan con un reloj falso que solo avanza dentro de los detectores y asesores
falsos; git y psutil también son falsos, salvo en las pruebas marcadas como "reales".
"""

import json
import platform
from contextlib import nullcontext
from pathlib import Path

import cv2
import numpy as np
import pytest

import src.pipeline.run as run
from src.data.schema import GROUPS, NO_VISIBLE
from src.pipeline.assessors.null import NullAssessor
from src.pipeline.detectors import DETECTORS
from src.pipeline.detectors.owlvit import Cargado, OwlViTConfig, OwlViTDetector
from src.pipeline.detectors.whole import WholeDetector
from src.pipeline.run import crear_parser, main, procesar_carpeta, procesar_imagen
from src.pipeline.types import Detection

CLAVES_ANTIGUAS = {"image", "lettuces", "timing_ms", "env"}
CLAVES_NUEVAS = {"image_size", "model_load_ms", "rss_mb"}
TIMING_ANTIGUO = {"detect", "assess", "total"}
TIMING_NUEVO = {"repeticiones", "calentamiento"}
CLAVES_LECHUGA = {"box", "score", "groups", "global", "provisional_labels"}
ENV_FALSO = {
    "python": platform.python_version(),
    "opencv": cv2.__version__,
    "torch": "2.14.1+cpu",
    "sistema": "Sistema 1",
    "device": "cpu",
    "detector": "whole",
    "commit": "abc1234",
    "dirty": False,
}


class RelojFalso:
    """Reloj que solo avanza cuando un detector o asesor falso lo dice."""

    def __init__(self):
        self.segundos = 0.0

    def __call__(self) -> float:
        return self.segundos

    def avanzar(self, milisegundos: float) -> None:
        self.segundos += milisegundos / 1000


class DetectorLento:
    """Detector falso: en cada llamada "tarda" la siguiente duración y devuelve ``n`` cajas."""

    def __init__(self, reloj, duraciones, n=1):
        self.reloj, self.pendientes, self.n, self.llamadas = reloj, iter(duraciones), n, 0

    def detect(self, imagen):
        self.llamadas += 1
        self.reloj.avanzar(next(self.pendientes))
        alto, ancho = imagen.shape[:2]
        return [Detection((0, 0, ancho, alto), 1.0) for _ in range(self.n)]


class AsesorLento(NullAssessor):
    """Asesor falso: cada evaluación "tarda" la siguiente duración (o una fija)."""

    def __init__(self, reloj, duraciones=None, fija=0.0):
        self.reloj, self.fija = reloj, fija
        self.pendientes = iter(duraciones) if duraciones is not None else None

    def assess(self, imagen, deteccion):
        self.reloj.avanzar(next(self.pendientes) if self.pendientes is not None else self.fija)
        return super().assess(imagen, deteccion)


class Contador:
    """Detector falso que cuenta cuántas veces se le llama."""

    def __init__(self, dispositivo=None, tiempo_carga_s=None):
        self.llamadas = 0
        if dispositivo is not None:
            self.dispositivo = dispositivo
        if tiempo_carga_s is not None:
            self.tiempo_carga_s = tiempo_carga_s

    def detect(self, imagen):
        self.llamadas += 1
        alto, ancho = imagen.shape[:2]
        return [Detection((0, 0, ancho, alto), 1.0)]


def _json(carpeta, nombre="a"):
    return json.loads((carpeta / f"{nombre}.json").read_text(encoding="utf-8"))


def _entrada(tmp_path, crear_imagen, escribir_imagen, nombres=("a.png",), alto=48, ancho=64):
    carpeta = tmp_path / "entrada"
    for nombre in nombres:
        escribir_imagen(carpeta / nombre, crear_imagen(alto, ancho))
    return carpeta


@pytest.fixture
def get_env_falso(monkeypatch):
    """Sustituye get_env en run.py por uno que anota con qué se le llama."""
    llamadas = []

    def falso(detector, dispositivo, **kwargs):
        llamadas.append((detector, dispositivo))
        return {**ENV_FALSO, "detector": detector, "device": dispositivo}

    monkeypatch.setattr(run, "get_env", falso)
    return llamadas


# --- Medianas dentro del pipeline (tiempos simulados) -----------------------------------------


def test_detect_y_assess_son_medianas_sin_las_pasadas_de_calentamiento(crear_imagen, tabla):
    reloj = RelojFalso()
    detector = DetectorLento(reloj, [100, 10, 30, 20])  # la primera es el calentamiento
    asesor = AsesorLento(reloj, [7, 1, 3, 2])

    resultado = procesar_imagen(
        crear_imagen(), "f.png", detector, asesor, tabla, repeticiones=3, calentamiento=1,
        reloj=reloj,
    )  # fmt: skip

    tiempos = resultado["timing_ms"]
    assert detector.llamadas == 4  # 1 de calentamiento + 3 medidas
    assert tiempos["detect"] == pytest.approx(20)  # si contase el calentamiento: 25
    assert tiempos["assess"] == pytest.approx(2)  # si contase el calentamiento: 2.5
    assert tiempos["total"] == pytest.approx(22)
    assert (tiempos["repeticiones"], tiempos["calentamiento"]) == (3, 1)


def test_total_es_la_mediana_de_los_ciclos_completos_y_no_la_suma_de_medianas(
    crear_imagen, tabla
):
    reloj = RelojFalso()
    detector = DetectorLento(reloj, [10, 20, 60])
    asesor = AsesorLento(reloj, [60, 20, 10])  # ciclos: 70, 40 y 70 ms

    tiempos = procesar_imagen(
        crear_imagen(), "f.png", detector, asesor, tabla, repeticiones=3, calentamiento=0,
        reloj=reloj,
    )["timing_ms"]  # fmt: skip

    assert tiempos["detect"] == pytest.approx(20) and tiempos["assess"] == pytest.approx(20)
    assert tiempos["total"] == pytest.approx(70)  # y no 40


def test_con_varias_detecciones_assess_suma_la_evaluacion_de_cada_una(crear_imagen, tabla):
    reloj = RelojFalso()
    detector = DetectorLento(reloj, [5, 5], n=2)
    asesor = AsesorLento(reloj, fija=2.0)  # 2 ms por lechuga

    resultado = procesar_imagen(
        crear_imagen(), "f.png", detector, asesor, tabla, repeticiones=2, calentamiento=0,
        reloj=reloj,
    )  # fmt: skip

    assert len(resultado["lettuces"]) == 2  # las lechugas salen de la última pasada, no se duplican
    assert resultado["timing_ms"]["assess"] == pytest.approx(4)
    assert resultado["timing_ms"]["total"] == pytest.approx(9)


def test_por_defecto_procesar_imagen_hace_una_sola_pasada_como_antes(crear_imagen, tabla):
    reloj = RelojFalso()
    detector = DetectorLento(reloj, [12])

    resultado = procesar_imagen(
        crear_imagen(), "f.png", detector, NullAssessor(), tabla, reloj=reloj
    )

    assert detector.llamadas == 1
    assert resultado["timing_ms"]["detect"] == pytest.approx(12)
    assert resultado["timing_ms"]["repeticiones"] == 1
    assert resultado["timing_ms"]["calentamiento"] == 0


def test_sin_detecciones_en_ninguna_pasada_no_falla(crear_imagen, tabla):
    reloj = RelojFalso()
    detector = DetectorLento(reloj, [1, 1, 1], n=0)

    resultado = procesar_imagen(
        crear_imagen(), "f.png", detector, NullAssessor(), tabla, repeticiones=2, calentamiento=1,
        reloj=reloj,
    )  # fmt: skip

    assert resultado["lettuces"] == [] and resultado["timing_ms"]["assess"] == 0


# --- Memoria, tamaño y carga del modelo -------------------------------------------------------


def test_rss_mb_se_mide_despues_de_todas_las_repeticiones(crear_imagen, tabla, monkeypatch):
    reloj = RelojFalso()
    detector = DetectorLento(reloj, [1, 1, 1, 1])
    monkeypatch.setattr(run, "rss_mb", lambda: float(detector.llamadas))  # "memoria" = llamadas

    resultado = procesar_imagen(
        crear_imagen(), "f.png", detector, NullAssessor(), tabla, repeticiones=3, calentamiento=1,
        reloj=reloj,
    )  # fmt: skip

    assert resultado["rss_mb"] == 4.0  # se ha leído cuando ya se habían hecho las 4 pasadas


@pytest.mark.parametrize(("alto", "ancho"), [(48, 64), (100, 30)])
def test_image_size_es_ancho_y_alto(crear_imagen, tabla, alto, ancho):
    resultado = procesar_imagen(
        crear_imagen(alto, ancho), "f.png", WholeDetector(), NullAssessor(), tabla
    )

    assert resultado["image_size"] == [ancho, alto]
    assert all(type(lado) is int for lado in resultado["image_size"])


def test_caso9_model_load_ms_es_null_con_el_detector_whole(crear_imagen, tabla):
    resultado = procesar_imagen(crear_imagen(), "f.png", WholeDetector(), NullAssessor(), tabla)

    assert "model_load_ms" in resultado and resultado["model_load_ms"] is None


def test_caso9_model_load_ms_es_un_numero_con_un_detector_que_carga_modelo(crear_imagen, tabla):
    detector = Contador(tiempo_carga_s=0.25)

    resultado = procesar_imagen(crear_imagen(), "f.png", detector, NullAssessor(), tabla)

    assert resultado["model_load_ms"] == pytest.approx(250.0)


def test_una_carga_de_cero_segundos_no_se_confunde_con_sin_modelo(crear_imagen, tabla):
    resultado = procesar_imagen(
        crear_imagen(), "f.png", Contador(tiempo_carga_s=0.0), NullAssessor(), tabla
    )

    assert resultado["model_load_ms"] == 0.0


def test_owlvit_expone_su_dispositivo_como_atributo_publico():
    config = OwlViTConfig("m", "q", 0.1, 5, 0.5, "cpu")

    def cargador(cfg):
        return Cargado(object(), object(), "cuda", nullcontext)

    assert OwlViTDetector(config, cargador=cargador).dispositivo == "cuda"


# --- Caso 7: claves nuevas y claves antiguas --------------------------------------------------


def test_caso7_procesar_imagen_trae_las_claves_nuevas_y_conserva_las_anteriores(
    crear_imagen, tabla
):
    resultado = procesar_imagen(crear_imagen(), "f.png", WholeDetector(), NullAssessor(), tabla)

    assert set(resultado) == CLAVES_ANTIGUAS | CLAVES_NUEVAS
    assert set(resultado["timing_ms"]) == TIMING_ANTIGUO | TIMING_NUEVO
    assert set(resultado["lettuces"][0]) == CLAVES_LECHUGA  # las lechugas, sin cambios
    assert resultado["image"] == "f.png"
    assert {"python", "opencv"} <= set(resultado["env"])
    assert isinstance(resultado["rss_mb"], float) and resultado["rss_mb"] > 0


def test_sin_env_se_usa_el_minimo_de_antes(crear_imagen, tabla):
    resultado = procesar_imagen(crear_imagen(), "f.png", WholeDetector(), NullAssessor(), tabla)

    assert resultado["env"] == {"python": platform.python_version(), "opencv": cv2.__version__}


def test_el_env_que_se_pasa_se_usa_tal_cual(crear_imagen, tabla):
    resultado = procesar_imagen(
        crear_imagen(), "f.png", WholeDetector(), NullAssessor(), tabla, env=ENV_FALSO
    )

    assert resultado["env"] == ENV_FALSO


def test_caso7_el_json_escrito_trae_las_claves_nuevas_y_las_antiguas(
    tmp_path, crear_imagen, escribir_imagen, tabla
):
    entrada = _entrada(tmp_path, crear_imagen, escribir_imagen)
    salida = tmp_path / "salida"

    procesar_carpeta(
        entrada, salida, WholeDetector(), NullAssessor(), tabla,
        env=ENV_FALSO, repeticiones=2, calentamiento=1,
    )  # fmt: skip

    datos = _json(salida)
    assert set(datos) == CLAVES_ANTIGUAS | CLAVES_NUEVAS
    assert datos["env"] == ENV_FALSO
    assert datos["image_size"] == [64, 48] and datos["model_load_ms"] is None
    assert (datos["timing_ms"]["repeticiones"], datos["timing_ms"]["calentamiento"]) == (2, 1)
    assert "daño_fisico" in datos["lettuces"][0]["groups"]  # sigue en UTF-8 con la ñ
    assert set(datos["lettuces"][0]["groups"]) == set(GROUPS)
    assert set(datos["lettuces"][0]["groups"].values()) == {NO_VISIBLE}


def test_caso7_via_main_con_get_env_y_rss_simulados(
    tmp_path, crear_imagen, escribir_imagen, get_env_falso, monkeypatch
):
    monkeypatch.setattr(run, "rss_mb", lambda: 123.4)
    entrada = _entrada(tmp_path, crear_imagen, escribir_imagen, ("a.png", "b.png"))
    salida = tmp_path / "salida"

    codigo = main(["--input", str(entrada), "--out", str(salida), "--detector", "whole"])

    datos = _json(salida, "b")
    assert codigo == 0
    assert set(datos) == CLAVES_ANTIGUAS | CLAVES_NUEVAS
    assert datos["rss_mb"] == 123.4
    assert datos["env"] == {**ENV_FALSO, "detector": "whole", "device": "cpu"}
    assert get_env_falso == [("whole", "cpu")]  # una sola vez, aunque hay dos imágenes


def test_get_env_recibe_el_dispositivo_que_declara_el_detector(
    tmp_path, crear_imagen, escribir_imagen, get_env_falso, monkeypatch
):
    monkeypatch.setitem(DETECTORS, "contador", lambda cfg: Contador(dispositivo="cuda"))
    entrada = _entrada(tmp_path, crear_imagen, escribir_imagen)

    main(["--input", str(entrada), "--out", str(tmp_path / "s"), "--detector", "contador"])

    assert get_env_falso == [("contador", "cuda")]
    assert _json(tmp_path / "s")["env"]["device"] == "cuda"


def test_via_main_con_las_funciones_reales_el_json_trae_el_env_completo(
    tmp_path, crear_imagen, escribir_imagen
):
    entrada = _entrada(tmp_path, crear_imagen, escribir_imagen)

    codigo = main(["--input", str(entrada), "--out", str(tmp_path / "s"), "--detector", "whole"])

    datos = _json(tmp_path / "s")
    assert codigo == 0
    assert list(datos["env"]) == [
        "python", "opencv", "torch", "sistema", "device", "detector", "commit", "dirty",
    ]  # fmt: skip
    assert datos["env"]["detector"] == "whole" and datos["env"]["device"] == "cpu"
    assert isinstance(datos["rss_mb"], float) and datos["rss_mb"] > 0
    assert Path(".").resolve().as_posix() not in json.dumps(datos)  # sin rutas absolutas


# --- Caso 8: --repeticiones y --calentamiento -------------------------------------------------


def test_los_valores_por_defecto_de_la_linea_de_comandos():
    argumentos = crear_parser().parse_args(["--input", "fotos"])

    assert argumentos.repeticiones == 5 and argumentos.calentamiento == 1


def test_caso8_con_repeticiones_1_y_calentamiento_0_se_mide_una_sola_vez(
    tmp_path, crear_imagen, escribir_imagen, get_env_falso, monkeypatch
):
    contador = Contador()
    monkeypatch.setitem(DETECTORS, "contador", lambda cfg: contador)
    entrada = _entrada(tmp_path, crear_imagen, escribir_imagen, ("a.png", "b.png"))

    main(["--input", str(entrada), "--out", str(tmp_path / "s"), "--detector", "contador",
          "--repeticiones", "1", "--calentamiento", "0"])  # fmt: skip

    assert contador.llamadas == 2  # una por imagen: el comportamiento de antes
    timing = _json(tmp_path / "s")["timing_ms"]
    assert (timing["repeticiones"], timing["calentamiento"]) == (1, 0)


def test_con_los_valores_por_defecto_se_ejecutan_seis_pasadas_por_imagen(
    tmp_path, crear_imagen, escribir_imagen, get_env_falso, monkeypatch
):
    contador = Contador()
    monkeypatch.setitem(DETECTORS, "contador", lambda cfg: contador)
    entrada = _entrada(tmp_path, crear_imagen, escribir_imagen, ("a.png", "b.png"))

    main(["--input", str(entrada), "--out", str(tmp_path / "s"), "--detector", "contador"])

    assert contador.llamadas == 12  # (1 de calentamiento + 5 repeticiones) x 2 imágenes
    timing = _json(tmp_path / "s")["timing_ms"]
    assert (timing["repeticiones"], timing["calentamiento"]) == (5, 1)


def test_las_opciones_se_pasan_tal_cual(
    tmp_path, crear_imagen, escribir_imagen, get_env_falso, monkeypatch
):
    contador = Contador()
    monkeypatch.setitem(DETECTORS, "contador", lambda cfg: contador)
    entrada = _entrada(tmp_path, crear_imagen, escribir_imagen)

    main(["--input", str(entrada), "--out", str(tmp_path / "s"), "--detector", "contador",
          "--repeticiones", "3", "--calentamiento", "2"])  # fmt: skip

    assert contador.llamadas == 5 and _json(tmp_path / "s")["timing_ms"]["repeticiones"] == 3


@pytest.mark.parametrize(
    ("opcion", "valor"),
    [("--repeticiones", "0"), ("--repeticiones", "-2"), ("--calentamiento", "-1")],
)
def test_valores_invalidos_dan_codigo_2_antes_de_cargar_el_modelo(
    tmp_path, crear_imagen, escribir_imagen, capsys, monkeypatch, opcion, valor
):
    creados = []
    monkeypatch.setitem(DETECTORS, "contador", lambda cfg: creados.append(1) or Contador())
    entrada = _entrada(tmp_path, crear_imagen, escribir_imagen)

    codigo = main(["--input", str(entrada), "--out", str(tmp_path / "s"), "--detector", "contador",
                   opcion, valor])  # fmt: skip

    assert codigo == 2 and opcion.lstrip("-") in capsys.readouterr().err
    assert creados == []  # no se llegó a crear (cargar) el detector


def test_un_valor_que_no_es_un_entero_lo_rechaza_el_parser(tmp_path):
    with pytest.raises(SystemExit) as salida:
        main(["--input", str(tmp_path), "--repeticiones", "abc"])

    assert salida.value.code == 2


def test_la_ayuda_avisa_del_coste_de_repetir(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])

    ayuda = capsys.readouterr().out
    assert "--repeticiones" in ayuda and "--calentamiento" in ayuda


def test_una_imagen_ilegible_sigue_saltandose_con_repeticiones(
    tmp_path, crear_imagen, escribir_imagen, get_env_falso
):
    entrada = _entrada(tmp_path, crear_imagen, escribir_imagen)
    (entrada / "rota.jpg").write_bytes(b"basura")

    with pytest.warns(UserWarning, match="rota.jpg"):
        codigo = main(["--input", str(entrada), "--out", str(tmp_path / "s"),
                       "--detector", "whole", "--repeticiones", "2"])  # fmt: skip

    assert codigo == 0 and (tmp_path / "s" / "a.json").is_file()
    assert not (tmp_path / "s" / "rota.json").exists()


def test_el_arreglo_de_imagen_no_se_modifica_al_repetir(crear_imagen, tabla):
    imagen = crear_imagen()
    antes = np.array(imagen)

    procesar_imagen(
        imagen, "f.png", WholeDetector(), NullAssessor(), tabla, repeticiones=3, calentamiento=1
    )

    np.testing.assert_array_equal(imagen, antes)
