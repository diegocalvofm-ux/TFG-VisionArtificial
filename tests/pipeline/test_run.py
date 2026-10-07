"""Pruebas del pipeline de extremo a extremo (src/pipeline/run.py)."""

import json
import platform
import warnings
from pathlib import Path

import cv2
import numpy as np

from src.data.schema import GROUPS, NO_VISIBLE
from src.pipeline.assessors.null import NullAssessor
from src.pipeline.detectors import DETECTORS
from src.pipeline.detectors.whole import WholeDetector
from src.pipeline.run import crear_parser, main, procesar_carpeta, procesar_imagen
from src.pipeline.types import Detection

RAIZ = Path(__file__).resolve().parents[2]
CLAVES_LECHUGA = {"box", "score", "groups", "global", "provisional_labels"}


class DetectorVacio:
    """Detector falso que no encuentra ninguna lechuga."""

    def detect(self, imagen):
        return []


class DetectorDoble:
    """Detector falso que encuentra dos lechugas."""

    def detect(self, imagen):
        return [Detection((0, 0, 10, 10), 0.9), Detection((5, 5, 20, 20), 0.4)]


class AsesorFalso:
    """Asesor falso que devuelve siempre los grupos que se le indican."""

    def __init__(self, **cambios):
        self.grupos = {**dict.fromkeys(GROUPS, NO_VISIBLE), **cambios}

    def assess(self, imagen, deteccion):
        return dict(self.grupos)


def _leer_json(ruta):
    return json.loads(ruta.read_text(encoding="utf-8"))


def _decodificar(ruta):
    return cv2.imdecode(np.fromfile(ruta, dtype=np.uint8), cv2.IMREAD_COLOR)


def _carpeta_con_imagenes(tmp_path, crear_imagen, escribir_imagen, nombres=("a.png", "b.png")):
    entrada = tmp_path / "entrada"
    for nombre in nombres:
        escribir_imagen(entrada / nombre, crear_imagen())
    return entrada


# --- procesar_imagen: el resultado y su integración con aggregate -----------------------


def test_caso3_el_pipeline_da_global_review_para_una_imagen_cualquiera(crear_imagen, tabla):
    resultado = procesar_imagen(crear_imagen(), "foto.png", WholeDetector(), NullAssessor(), tabla)

    assert len(resultado["lettuces"]) == 1
    assert resultado["lettuces"][0]["global"] == "REVIEW"
    assert resultado["lettuces"][0]["provisional_labels"] == []


def test_el_pipeline_usa_aggregate_con_la_tabla(crear_imagen, tabla):
    # ROOT_REMAINS manda (NON-CONFORMING) y WILTING, aún PENDIENTE, queda como provisional.
    asesor = AsesorFalso(corte="ROOT_REMAINS", color="WILTING")

    resultado = procesar_imagen(crear_imagen(), "foto.png", WholeDetector(), asesor, tabla)

    lechuga = resultado["lettuces"][0]
    assert lechuga["global"] == "NON-CONFORMING"
    assert lechuga["provisional_labels"] == ["WILTING"]
    assert lechuga["groups"]["corte"] == "ROOT_REMAINS"


def test_cada_deteccion_se_evalua_por_separado(crear_imagen, tabla):
    resultado = procesar_imagen(crear_imagen(), "f.png", DetectorDoble(), NullAssessor(), tabla)

    assert [lechuga["box"] for lechuga in resultado["lettuces"]] == [[0, 0, 10, 10], [5, 5, 20, 20]]
    assert [lechuga["score"] for lechuga in resultado["lettuces"]] == [0.9, 0.4]


def test_caso6_sin_detecciones_da_lettuces_vacio_sin_fallar(crear_imagen, tabla):
    resultado = procesar_imagen(crear_imagen(), "f.png", DetectorVacio(), NullAssessor(), tabla)

    assert resultado["lettuces"] == []
    assert resultado["timing_ms"]["assess"] == 0


def test_caso4_la_estructura_del_resultado(crear_imagen, tabla):
    resultado = procesar_imagen(crear_imagen(), "foto.png", WholeDetector(), NullAssessor(), tabla)

    assert set(resultado) == {"image", "lettuces", "timing_ms", "env"}
    assert resultado["image"] == "foto.png"
    assert set(resultado["timing_ms"]) == {"detect", "assess", "total"}
    assert set(resultado["env"]) == {"python", "opencv"}
    assert resultado["env"]["python"] == platform.python_version()
    assert resultado["env"]["opencv"] == cv2.__version__
    lechuga = resultado["lettuces"][0]
    assert set(lechuga) == CLAVES_LECHUGA
    assert lechuga["box"] == [0, 0, 64, 48]
    assert all(type(valor) is int for valor in lechuga["box"])
    assert list(lechuga["groups"]) == list(GROUPS)


def test_caso11_los_tiempos_son_numeros_no_negativos(crear_imagen, tabla):
    resultado = procesar_imagen(crear_imagen(), "foto.png", WholeDetector(), NullAssessor(), tabla)
    tiempos = resultado["timing_ms"]

    for valor in tiempos.values():
        assert isinstance(valor, float) and valor >= 0
    assert tiempos["total"] >= tiempos["detect"] + tiempos["assess"] - 1e-9


# --- procesar_carpeta: archivos de salida -----------------------------------------------


def test_caso4_el_json_escrito_tiene_el_esquema_y_se_relee(
    tmp_path, crear_imagen, escribir_imagen, tabla
):
    entrada = _carpeta_con_imagenes(tmp_path, crear_imagen, escribir_imagen, ("foto01.png",))
    salida = tmp_path / "salida"

    procesar_carpeta(entrada, salida, WholeDetector(), NullAssessor(), tabla)

    ruta = salida / "foto01.json"
    datos = _leer_json(ruta)
    assert set(datos) == {"image", "lettuces", "timing_ms", "env"}
    assert datos["image"] == "foto01.png"
    assert set(datos["lettuces"][0]) == CLAVES_LECHUGA
    assert list(datos["lettuces"][0]["groups"]) == list(GROUPS)
    assert datos["lettuces"][0]["global"] == "REVIEW"
    # Con ensure_ascii=False la ñ se guarda como UTF-8 real, no como ñ.
    crudo = ruta.read_bytes()
    assert "daño_fisico".encode("utf-8") in crudo
    assert b"\\u00f1" not in crudo


def test_caso5_la_imagen_anotada_se_crea_y_es_distinta_de_la_original(
    tmp_path, crear_imagen, escribir_imagen, tabla
):
    original = crear_imagen()
    entrada = tmp_path / "entrada"
    escribir_imagen(entrada / "foto.png", original)
    salida = tmp_path / "salida"

    procesar_carpeta(entrada, salida, WholeDetector(), NullAssessor(), tabla)

    anotada = _decodificar(salida / "foto_anotada.jpg")
    # Se compara con la original tras el mismo ciclo JPEG: así la diferencia es el dibujo
    # y no solo la compresión.
    _, datos = cv2.imencode(".jpg", original)
    original_jpg = cv2.imdecode(datos, cv2.IMREAD_COLOR)
    assert anotada.shape == original.shape
    diferencia = np.abs(anotada.astype(int) - original_jpg.astype(int))
    assert diferencia.max() > 50


def test_caso6_sin_detecciones_se_escriben_json_e_imagen_sin_dibujo(
    tmp_path, crear_imagen, escribir_imagen, tabla
):
    original = crear_imagen()
    entrada = tmp_path / "entrada"
    escribir_imagen(entrada / "foto.png", original)
    salida = tmp_path / "salida"

    procesadas, saltadas = procesar_carpeta(entrada, salida, DetectorVacio(), NullAssessor(), tabla)

    assert (procesadas, saltadas) == (1, 0)
    assert _leer_json(salida / "foto.json")["lettuces"] == []
    _, datos = cv2.imencode(".jpg", original)
    np.testing.assert_array_equal(
        _decodificar(salida / "foto_anotada.jpg"), cv2.imdecode(datos, cv2.IMREAD_COLOR)
    )


def test_caso7_un_archivo_corrupto_se_salta_con_aviso_y_el_resto_se_procesa(
    tmp_path, crear_imagen, escribir_imagen, tabla
):
    entrada = _carpeta_con_imagenes(tmp_path, crear_imagen, escribir_imagen, ("ok1.png", "ok2.png"))
    (entrada / "corrupta.jpg").write_bytes(b"esto no es una imagen")
    (entrada / "vacia.png").write_bytes(b"")
    (entrada / "notas.txt").write_text("no soy una imagen", encoding="utf-8")
    salida = tmp_path / "salida"

    with warnings.catch_warnings(record=True) as avisos:
        warnings.simplefilter("always")
        procesadas, saltadas = procesar_carpeta(
            entrada, salida, WholeDetector(), NullAssessor(), tabla
        )

    assert (procesadas, saltadas) == (2, 3)
    textos = [str(aviso.message) for aviso in avisos if issubclass(aviso.category, UserWarning)]
    for nombre in ("corrupta.jpg", "vacia.png", "notas.txt"):
        assert any(nombre in texto for texto in textos), nombre
    assert {p.name for p in salida.iterdir()} == {
        "ok1.json",
        "ok1_anotada.jpg",
        "ok2.json",
        "ok2_anotada.jpg",
    }


def test_las_extensiones_se_admiten_sin_distinguir_mayusculas(
    tmp_path, crear_imagen, escribir_imagen, tabla
):
    entrada = tmp_path / "entrada"
    escribir_imagen(entrada / "FOTO.PNG", crear_imagen())
    escribir_imagen(entrada / "otra.bmp", crear_imagen())
    escribir_imagen(entrada / "tercera.jpeg", crear_imagen())

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        procesadas, saltadas = procesar_carpeta(
            entrada, tmp_path / "salida", WholeDetector(), NullAssessor(), tabla
        )

    assert (procesadas, saltadas) == (3, 0)


def test_dos_archivos_con_el_mismo_nombre_base_no_se_sobrescriben(
    tmp_path, crear_imagen, escribir_imagen, tabla
):
    nombres = ("foto.jpg", "foto.png")
    entrada = _carpeta_con_imagenes(tmp_path, crear_imagen, escribir_imagen, nombres)
    salida = tmp_path / "salida"

    with warnings.catch_warnings(record=True) as avisos:
        warnings.simplefilter("always")
        procesadas, saltadas = procesar_carpeta(
            entrada, salida, WholeDetector(), NullAssessor(), tabla
        )

    assert (procesadas, saltadas) == (1, 1)
    assert any("foto.png" in str(aviso.message) for aviso in avisos)
    assert _leer_json(salida / "foto.json")["image"] == "foto.jpg"


def test_las_subcarpetas_se_ignoran_sin_avisos(tmp_path, crear_imagen, escribir_imagen, tabla):
    entrada = _carpeta_con_imagenes(tmp_path, crear_imagen, escribir_imagen, ("a.png",))
    escribir_imagen(entrada / "sub" / "dentro.png", crear_imagen())

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        procesadas, saltadas = procesar_carpeta(
            entrada, tmp_path / "salida", WholeDetector(), NullAssessor(), tabla
        )

    assert (procesadas, saltadas) == (1, 0)


def test_caso8_nombres_con_tildes_y_enes_en_archivo_y_carpetas(
    tmp_path, crear_imagen, escribir_imagen, tabla
):
    entrada = tmp_path / "entrada ñ"
    escribir_imagen(entrada / "lechuga_año_ñandú.png", crear_imagen())
    salida = tmp_path / "salida_ñ"

    procesadas, saltadas = procesar_carpeta(
        entrada, salida, WholeDetector(), NullAssessor(), tabla
    )

    assert (procesadas, saltadas) == (1, 0)
    assert {p.name for p in salida.iterdir()} == {
        "lechuga_año_ñandú.json",
        "lechuga_año_ñandú_anotada.jpg",
    }
    assert _leer_json(salida / "lechuga_año_ñandú.json")["image"] == "lechuga_año_ñandú.png"
    assert _decodificar(salida / "lechuga_año_ñandú_anotada.jpg") is not None


# --- main(argv) --------------------------------------------------------------------------


def test_los_valores_por_defecto_salen_de_la_raiz_del_repositorio():
    argumentos = crear_parser().parse_args(["--input", "fotos"])

    assert argumentos.out == RAIZ / "runs" / "v0"
    assert argumentos.config == RAIZ / "configs" / "v0.yaml"
    assert argumentos.tabla == RAIZ / "configs" / "salida_global.yaml"
    assert argumentos.detector is None  # se toma del archivo de configuración


def test_caso10_main_con_dos_imagenes_validas_y_una_corrupta(
    tmp_path, crear_imagen, escribir_imagen, capsys
):
    entrada = _carpeta_con_imagenes(tmp_path, crear_imagen, escribir_imagen, ("a.png", "c.png"))
    (entrada / "b_corrupta.jpg").write_bytes(b"basura")
    salida = tmp_path / "salida"

    with warnings.catch_warnings(record=True) as avisos:
        warnings.simplefilter("always")
        codigo = main(["--input", str(entrada), "--out", str(salida), "--detector", "whole"])

    assert codigo == 0
    assert sorted(p.name for p in salida.glob("*.json")) == ["a.json", "c.json"]
    anotadas = sorted(p.name for p in salida.glob("*_anotada.jpg"))
    assert anotadas == ["a_anotada.jpg", "c_anotada.jpg"]
    assert any("b_corrupta.jpg" in str(aviso.message) for aviso in avisos)
    resumen = capsys.readouterr().out
    assert "Procesadas: 2" in resumen and "Saltadas: 1" in resumen


def test_main_sin_detector_usa_el_de_la_configuracion(
    tmp_path, crear_imagen, escribir_imagen
):
    entrada = _carpeta_con_imagenes(tmp_path, crear_imagen, escribir_imagen, ("a.png",))
    salida = tmp_path / "salida"

    codigo = main(["--input", str(entrada), "--out", str(salida)])

    assert codigo == 0
    assert (salida / "a.json").is_file()


def test_main_con_un_detector_registrado_sin_detecciones(
    tmp_path, crear_imagen, escribir_imagen, monkeypatch
):
    monkeypatch.setitem(DETECTORS, "vacio", lambda cfg: DetectorVacio())
    entrada = _carpeta_con_imagenes(tmp_path, crear_imagen, escribir_imagen, ("a.png",))
    salida = tmp_path / "salida"

    codigo = main(["--input", str(entrada), "--out", str(salida), "--detector", "vacio"])

    assert codigo == 0
    assert _leer_json(salida / "a.json")["lettuces"] == []


def test_caso9_main_con_un_detector_desconocido_da_error_claro(
    tmp_path, crear_imagen, escribir_imagen, capsys
):
    entrada = _carpeta_con_imagenes(tmp_path, crear_imagen, escribir_imagen, ("a.png",))
    salida = tmp_path / "salida"

    codigo = main(["--input", str(entrada), "--out", str(salida), "--detector", "nope"])

    assert codigo == 2
    error = capsys.readouterr().err
    assert "nope" in error and "whole" in error
    assert not salida.exists() or not list(salida.iterdir())


def test_main_con_carpeta_vacia_devuelve_1(tmp_path, capsys):
    entrada = tmp_path / "vacia"
    entrada.mkdir()

    codigo = main(["--input", str(entrada), "--out", str(tmp_path / "salida")])

    assert codigo == 1
    assert "ninguna" in capsys.readouterr().err.lower()


def test_main_con_carpeta_inexistente_devuelve_2(tmp_path, capsys):
    codigo = main(["--input", str(tmp_path / "no_existe"), "--out", str(tmp_path / "salida")])

    assert codigo == 2
    assert "no_existe" in capsys.readouterr().err


def test_main_con_configuracion_inexistente_devuelve_2(
    tmp_path, crear_imagen, escribir_imagen, capsys
):
    entrada = _carpeta_con_imagenes(tmp_path, crear_imagen, escribir_imagen, ("a.png",))

    inexistente = tmp_path / "x.yaml"

    argv = ["--input", str(entrada), "--out", str(tmp_path / "s"), "--config", str(inexistente)]

    codigo = main(argv)

    assert codigo == 2
    assert "x.yaml" in capsys.readouterr().err
