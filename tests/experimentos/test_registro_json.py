"""Registro con los datos que escribe el pipeline desde la tarea 5 (y con los JSON antiguos).

Todo lo que se guarda en la nota sale del JSON: ``env.detector``, ``env.device``, ``env.commit``,
``env.dirty``, ``image_size``, ``rss_mb``, ``model_load_ms`` y ``timing_ms`` (con
``repeticiones`` y ``calentamiento``). Si un JSON no trae algo, la celda queda vacía y se avisa.
"""

import re
import shutil
import subprocess
from datetime import date
from pathlib import Path

import pytest

from src.experimentos.ejecuciones import parsear_tabla
from src.experimentos.nota import leer_bloque, leer_cabecera
from src.experimentos.registro import main

HOY = date(2026, 10, 7)
RAIZ = Path(__file__).resolve().parents[2]
CAMPOS_DEL_JSON = (
    "env.detector", "env.device", "env.commit", "env.dirty", "image_size", "rss_mb",
    "model_load_ms", "timing_ms.repeticiones", "timing_ms.calentamiento",
)  # fmt: skip


@pytest.fixture(autouse=True)
def sin_subprocesos(monkeypatch):
    """El registro ya no consulta git: si lo intentara, la prueba falla."""

    def prohibido(*args, **kwargs):
        raise AssertionError("el registro no debe lanzar procesos (git)")

    monkeypatch.setattr(subprocess, "run", prohibido)


def ejecutar(docs, *args):
    return main(["--docs", str(docs.raiz), *args], hoy=HOY)


def _texto(docs, exp="exp001"):
    return (docs.carpeta / f"{exp}.md").read_text(encoding="utf-8")


def _cabecera(docs, exp="exp001"):
    return leer_cabecera(_texto(docs, exp))


def _filas(docs, exp="exp001"):
    return parsear_tabla(leer_bloque(_texto(docs, exp), "resultados") or "")


def _runs(tmp_path, escribir_json, nombre="runs", n=3, **opciones):
    for i in range(n):
        escribir_json(tmp_path / nombre, f"f{i}", total_ms=10 * (i + 1), **opciones)
    return tmp_path / nombre


def _ejecucion(docs, runs, *extra):
    return ejecutar(docs, "ejecucion", "--exp", "exp001", "--runs", str(runs), *extra)


# --- JSON completo ----------------------------------------------------------------------


def test_json_completo_rellena_la_fila_entera_y_no_avisa_de_nada(
    docs, escribir_json, tmp_path, capsys
):
    ejecutar(docs, "nuevo")
    capsys.readouterr()
    runs = _runs(
        tmp_path, escribir_json, completo=True, env={"detector": "owlvit", "device": "cuda"},
        raiz={"model_load_ms": 8523.5, "rss_mb": 1020.57, "image_size": [527, 612]},
    )  # fmt: skip

    codigo = _ejecucion(docs, runs)

    fila = _filas(docs)[0]
    salida = capsys.readouterr()
    assert codigo == 0
    assert (fila["Detector"], fila["Dispositivo"]) == ("owlvit", "cuda")
    assert (fila["Commit"], fila["Sucio"]) == ("abc1234", "no")
    assert fila["Tamaño imagen"] == "527x612"  # [ancho, alto], tal como lo escribe el JSON
    assert (fila["Repeticiones"], fila["Calentamiento"]) == ("5", "1")
    assert fila["Carga modelo (ms)"] == "8523.5" and fila["Memoria (MB)"] == "1020.6"
    assert salida.err == ""  # no falta nada, no hay árbol sucio: ningún aviso


def test_con_el_detector_whole_model_load_ms_es_null_y_no_se_avisa(
    docs, escribir_json, tmp_path, capsys
):
    ejecutar(docs, "nuevo")
    capsys.readouterr()

    _ejecucion(docs, _runs(tmp_path, escribir_json, completo=True))  # model_load_ms: null

    assert _filas(docs)[0]["Carga modelo (ms)"] == ""
    assert "model_load_ms" not in capsys.readouterr().err


# --- JSON antiguo (anterior a la tarea 5) -----------------------------------------------


def test_json_antiguo_queda_vacio_y_se_avisa_de_cada_dato_que_falta(
    docs, escribir_json, tmp_path, capsys
):
    ejecutar(docs, "nuevo")
    capsys.readouterr()
    runs = _runs(tmp_path, escribir_json)  # como runs/v0_local: sin ninguno de los campos nuevos

    codigo = _ejecucion(docs, runs)

    fila = _filas(docs)[0]
    salida = capsys.readouterr()
    assert codigo == 0  # no es un error: se registra lo que hay
    for columna in ("Detector", "Dispositivo", "Commit", "Sucio", "Tamaño imagen", "Repeticiones",
                    "Calentamiento", "Carga modelo (ms)", "Memoria (MB)"):  # fmt: skip
        assert fila[columna] == "", columna  # vacío, nunca 0 ni inventado
    assert fila["Imágenes"] == "3" and fila["Mediana (ms)"] == "20.00"  # lo medible sí
    aviso = next(x for x in salida.err.splitlines() if "faltan datos" in x)
    for campo in CAMPOS_DEL_JSON:
        assert f"{campo} (3 de 3)" in aviso, campo
    assert "faltan datos" not in salida.out  # es un aviso: va por la salida de errores


def test_json_antiguo_deja_el_commit_de_la_cabecera_vacio_y_dice_como_ponerlo(
    docs, escribir_json, tmp_path, capsys
):
    ejecutar(docs, "nuevo")

    _ejecucion(docs, _runs(tmp_path, escribir_json))

    assert _cabecera(docs)["commit"] is None  # no se inventa ni se pregunta a git
    assert "--commit" in capsys.readouterr().err


def test_json_antiguo_con_commit_dado_a_mano_lo_guarda_en_la_cabecera(
    docs, escribir_json, tmp_path
):
    ejecutar(docs, "nuevo")

    _ejecucion(docs, _runs(tmp_path, escribir_json), "--commit", "def5678")

    assert _cabecera(docs)["commit"] == "def5678"
    assert _filas(docs)[0]["Commit"] == ""  # la fila solo lleva lo que mide el JSON


def test_json_mixtos_cuentan_los_que_faltan_por_campo(docs, escribir_json, tmp_path, capsys):
    ejecutar(docs, "nuevo")
    capsys.readouterr()
    runs = tmp_path / "runs"
    escribir_json(runs, "a", completo=True)
    escribir_json(runs, "b", completo=True, sin=("rss_mb", "image_size"))
    escribir_json(runs, "c")  # antiguo

    _ejecucion(docs, runs)

    aviso = next(x for x in capsys.readouterr().err.splitlines() if "faltan datos" in x)
    assert "rss_mb (2 de 3)" in aviso and "image_size (2 de 3)" in aviso
    assert "env.commit (1 de 3)" in aviso and "model_load_ms (1 de 3)" in aviso
    fila = _filas(docs)[0]
    assert fila["Detector"] == "whole" and fila["Memoria (MB)"] == "80.0"  # usa los que hay


def test_un_commit_o_dirty_a_null_cuentan_como_dato_que_falta(
    docs, escribir_json, tmp_path, capsys
):
    ejecutar(docs, "nuevo")
    capsys.readouterr()
    runs = _runs(tmp_path, escribir_json, n=2, completo=True, env={"commit": None, "dirty": None})

    _ejecucion(docs, runs)

    aviso = next(x for x in capsys.readouterr().err.splitlines() if "faltan datos" in x)
    assert "env.commit (2 de 2)" in aviso and "env.dirty (2 de 2)" in aviso
    assert (_filas(docs)[0]["Commit"], _filas(docs)[0]["Sucio"]) == ("", "")


def test_los_json_ignorados_y_los_campos_mal_formados_se_avisan(
    docs, escribir_json, tmp_path, capsys
):
    ejecutar(docs, "nuevo")
    capsys.readouterr()
    runs = _runs(tmp_path, escribir_json, n=2, completo=True)
    (runs / "rota.json").write_bytes(b"no es json")
    escribir_json(runs, "f9", completo=True, raiz={"rss_mb": "mucho"})

    codigo = _ejecucion(docs, runs)

    error = capsys.readouterr().err
    assert codigo == 0 and "rota.json" in error
    assert "f9.json" in error and "rss_mb" in error


# --- dirty ------------------------------------------------------------------------------


@pytest.mark.parametrize("extra", [(), ("--commit", "def5678")], ids=["sin_commit", "con_commit"])
def test_dirty_true_avisa_de_que_no_se_reproduce_solo_con_el_commit(
    docs, escribir_json, tmp_path, capsys, extra
):
    ejecutar(docs, "nuevo")
    capsys.readouterr()
    runs = _runs(tmp_path, escribir_json, completo=True, env={"dirty": True})

    _ejecucion(docs, runs, *extra)

    error = capsys.readouterr().err
    assert "sin commitear" in error and "no se reproduce solo con el commit" in error
    assert _filas(docs)[0]["Sucio"] == "sí"


def test_dirty_false_no_avisa(docs, escribir_json, tmp_path, capsys):
    ejecutar(docs, "nuevo")
    capsys.readouterr()

    _ejecucion(docs, _runs(tmp_path, escribir_json, completo=True))

    assert "sin commitear" not in capsys.readouterr().err


def test_si_algun_json_estaba_sucio_se_avisa_aunque_los_demas_no(
    docs, escribir_json, tmp_path, capsys
):
    ejecutar(docs, "nuevo")
    capsys.readouterr()
    runs = tmp_path / "runs"
    escribir_json(runs, "a", completo=True, env={"dirty": True})
    escribir_json(runs, "b", completo=True, env={"dirty": False})

    _ejecucion(docs, runs)

    assert "sin commitear" in capsys.readouterr().err
    assert _filas(docs)[0]["Sucio"] == ""  # mezclado: no se elige un valor


# --- commit -----------------------------------------------------------------------------


def test_commits_distintos_entre_json_avisan_y_no_se_escribe_ninguno(
    docs, escribir_json, tmp_path, capsys
):
    ejecutar(docs, "nuevo")
    capsys.readouterr()
    runs = tmp_path / "runs"
    escribir_json(runs, "a", completo=True, env={"commit": "abc1234"})
    escribir_json(runs, "b", completo=True, env={"commit": "def5678"})

    _ejecucion(docs, runs)

    error = capsys.readouterr().err
    assert "abc1234" in error and "def5678" in error
    assert _cabecera(docs)["commit"] is None and _filas(docs)[0]["Commit"] == ""


def test_el_commit_del_json_se_anota_en_la_fila_y_en_la_cabecera(docs, escribir_json, tmp_path):
    ejecutar(docs, "nuevo")

    _ejecucion(docs, _runs(tmp_path, escribir_json, completo=True, env={"commit": "806f649"}))

    assert _cabecera(docs)["commit"] == "806f649" and _filas(docs)[0]["Commit"] == "806f649"


# --- exp000 real ------------------------------------------------------------------------


def test_exp000_real_se_actualiza_sin_perder_su_texto(tmp_path, escribir_json):
    docs = tmp_path / "docs"
    (docs / "plantillas").mkdir(parents=True)
    (docs / "experimentos").mkdir()
    shutil.copy(RAIZ / "docs" / "plantillas" / "experimento.md", docs / "plantillas")
    shutil.copy(RAIZ / "docs" / "experimentos" / "exp000.md", docs / "experimentos")
    nota = docs / "experimentos" / "exp000.md"
    original = nota.read_text(encoding="utf-8")
    runs = _runs(tmp_path, escribir_json, completo=True)
    base = ["--docs", str(docs)]
    campos_tocados = ("commit:", "valor:", "metrica_principal:")

    assert main([*base, "ejecucion", "--exp", "exp000", "--runs", str(runs)], hoy=HOY) == 0
    cierre = ["campos", "--exp", "exp000", "--set", "valor=0.35"]
    cierre += ["--set", "metrica_principal=mAP50"]
    assert main([*base, *cierre], hoy=HOY) == 0

    nuevo = nota.read_text(encoding="utf-8")
    bloque = r"\n<!-- registro:resultados -->.*?<!-- /registro:resultados -->\n\n"
    sin_bloque = re.sub(bloque, "", nuevo, flags=re.DOTALL)
    assert sin_bloque != nuevo  # había bloque
    esperado = [x for x in original.splitlines() if not x.startswith(campos_tocados)]
    obtenido = [x for x in sin_bloque.splitlines() if not x.startswith(campos_tocados)]
    assert obtenido == esperado  # ni una línea de su texto cambia, salvo las tres de la cabecera
    cabecera = leer_cabecera(nuevo)
    assert cabecera["commit"] == "abc1234" and cabecera["valor"] == 0.35
    assert cabecera["metrica_principal"] == "mAP50"
    assert (cabecera["id"], cabecera["tags"]) == ("exp000", ["experimento"])
    assert cabecera["entorno"] == "colab-t4" and cabecera["torch"] == "2.11.0+cu130"
