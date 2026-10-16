"""Pruebas de src/utils/env.py: versiones, dispositivo, commit y estado de git del entorno."""

import getpass
import importlib.metadata
import json
import platform
import re
from pathlib import Path

import cv2
import pytest

from src.utils.env import get_env

CLAVES = ["python", "opencv", "torch", "sistema", "device", "detector", "commit", "dirty"]
USUARIO_FICTICIO = "usuario_prueba"
USUARIO_REAL = getpass.getuser()
RUTA_ABSOLUTA = re.compile(r"[A-Za-z]:[\\/]|(?<![\w/.:])/(?:home|Users|tmp|var|root|mnt|content)/")


def _seguro(valor) -> bool:
    """True si el valor no es un texto con ruta absoluta ni con algún nombre de usuario."""
    if not isinstance(valor, str):
        return True
    minusculas = valor.lower()
    nombres = [USUARIO_FICTICIO, USUARIO_REAL.lower()]
    return not RUTA_ABSOLUTA.search(valor) and not any(
        len(n) >= 3 and n in minusculas for n in nombres
    )


# --- Caso 3: las claves ---------------------------------------------------------------------


def test_caso3_get_env_devuelve_todas_las_claves_en_orden(tmp_path, git_falso):
    env = get_env("whole", "cpu", raiz=tmp_path, ejecutar=git_falso())

    assert list(env) == CLAVES


def test_los_valores_de_python_opencv_y_sistema_salen_de_la_plataforma(tmp_path, git_falso):
    env = get_env("whole", "cpu", raiz=tmp_path, ejecutar=git_falso())

    assert env["python"] == platform.python_version()
    assert env["opencv"] == cv2.__version__
    assert env["sistema"] == f"{platform.system()} {platform.release()}"


def test_sistema_es_solo_el_sistema_operativo_y_su_version(tmp_path, git_falso, monkeypatch):
    monkeypatch.setattr(platform, "system", lambda: "Windows")
    monkeypatch.setattr(platform, "release", lambda: "11")

    env = get_env("whole", "cpu", raiz=tmp_path, ejecutar=git_falso())

    assert env["sistema"] == "Windows 11"


def test_device_y_detector_son_los_que_se_piden(tmp_path, git_falso):
    env = get_env("owlvit", "cuda", raiz=tmp_path, ejecutar=git_falso())

    assert env["device"] == "cuda" and env["detector"] == "owlvit"


@pytest.mark.parametrize("dispositivo", ["tpu", "auto", "", "CPU"])
def test_un_dispositivo_que_no_es_cpu_ni_cuda_lanza_value_error(tmp_path, git_falso, dispositivo):
    with pytest.raises(ValueError, match="device"):
        get_env("whole", dispositivo, raiz=tmp_path, ejecutar=git_falso())


def test_caso3_torch_es_none_si_no_esta_instalado(tmp_path, git_falso, monkeypatch):
    def sin_torch(nombre):
        raise importlib.metadata.PackageNotFoundError(nombre)

    monkeypatch.setattr(importlib.metadata, "version", sin_torch)

    env = get_env("whole", "cpu", raiz=tmp_path, ejecutar=git_falso())

    assert env["torch"] is None
    assert list(env) == CLAVES  # y el resto de claves siguen ahí


def test_torch_es_la_version_instalada_sin_importarlo(tmp_path, git_falso, monkeypatch):
    pedidos = []

    def version_falsa(nombre):
        pedidos.append(nombre)
        return "2.14.1+cpu"

    monkeypatch.setattr(importlib.metadata, "version", version_falsa)

    env = get_env("whole", "cpu", raiz=tmp_path, ejecutar=git_falso())

    assert env["torch"] == "2.14.1+cpu" and pedidos == ["torch"]


def test_el_env_se_puede_guardar_como_json(tmp_path, git_falso):
    env = get_env("whole", "cpu", raiz=tmp_path, ejecutar=git_falso(status=" M src/x.py"))

    assert json.loads(json.dumps(env)) == env


# --- Caso 4: fuera de un repo o sin git --------------------------------------------------------


def test_caso4_fuera_de_un_repositorio_commit_y_dirty_son_none(tmp_path, git_falso):
    git = git_falso(codigo_rev=128, codigo_status=128)

    env = get_env("whole", "cpu", raiz=tmp_path, ejecutar=git)

    assert env["commit"] is None and env["dirty"] is None


def test_caso4_sin_git_instalado_commit_y_dirty_son_none_sin_error(tmp_path, git_falso):
    env = get_env("whole", "cpu", raiz=tmp_path, ejecutar=git_falso(falta=True))

    assert env["commit"] is None and env["dirty"] is None
    assert env["python"] == platform.python_version()  # lo demás sigue funcionando


def test_si_git_status_falla_pero_hay_commit_dirty_es_none(tmp_path, git_falso):
    env = get_env("whole", "cpu", raiz=tmp_path, ejecutar=git_falso(codigo_status=128))

    assert env["commit"] == "abc1234" and env["dirty"] is None


def test_un_commit_vacio_se_toma_como_no_disponible(tmp_path, git_falso):
    env = get_env("whole", "cpu", raiz=tmp_path, ejecutar=git_falso(commit=""))

    assert env["commit"] is None


# --- Caso 5: dirty -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "estado", [" M src/pipeline/run.py\n", "?? src/nuevo.py\n", "A  configs/x.yaml\n"]
)
def test_caso5_con_cambios_sin_commitear_en_src_o_configs_dirty_es_true(
    tmp_path, git_falso, estado
):
    env = get_env("whole", "cpu", raiz=tmp_path, ejecutar=git_falso(status=estado))

    assert env["dirty"] is True and env["commit"] == "abc1234"


def test_sin_cambios_dirty_es_false(tmp_path, git_falso):
    env = get_env("whole", "cpu", raiz=tmp_path, ejecutar=git_falso(status=""))

    assert env["dirty"] is False


def test_los_comandos_de_git_son_los_pedidos_y_van_a_la_raiz(tmp_path, git_falso):
    git = git_falso()

    get_env("whole", "cpu", raiz=tmp_path, ejecutar=git)

    assert ["git", "-C", str(tmp_path), "rev-parse", "--short", "HEAD"] in git.llamadas
    assert [
        "git", "-C", str(tmp_path), "status", "--porcelain", "--", "src", "configs"
    ] in git.llamadas  # fmt: skip


def test_por_defecto_la_raiz_es_la_del_repositorio(git_falso):
    git = git_falso()

    get_env("whole", "cpu", ejecutar=git)

    raiz = Path(__file__).resolve().parents[2]
    assert all(orden[:3] == ["git", "-C", str(raiz)] for orden in git.llamadas)


# --- Caso 6: privacidad ---------------------------------------------------------------------


@pytest.fixture
def usuario_ficticio(monkeypatch):
    monkeypatch.setattr(getpass, "getuser", lambda: USUARIO_FICTICIO)


def test_caso6_un_valor_con_el_usuario_se_sustituye_por_none(
    tmp_path, git_falso, monkeypatch, usuario_ficticio
):
    monkeypatch.setattr(platform, "release", lambda: f"10.0.{USUARIO_FICTICIO}")

    env = get_env("whole", "cpu", raiz=tmp_path, ejecutar=git_falso())

    assert env["sistema"] is None
    assert all(_seguro(v) for v in env.values())


@pytest.mark.parametrize(
    "detector", [r"C:\Users\ana\modelo.pt", "C:/Users/ana/modelo.pt", "/home/ana/modelo.pt"]
)
def test_caso6_un_valor_con_una_ruta_absoluta_se_sustituye_por_none(
    tmp_path, git_falso, usuario_ficticio, detector
):
    env = get_env(detector, "cpu", raiz=tmp_path, ejecutar=git_falso())

    assert env["detector"] is None
    assert all(_seguro(v) for v in env.values())


def test_caso6_un_commit_con_una_ruta_o_el_usuario_tambien_se_sustituye(
    tmp_path, git_falso, usuario_ficticio
):
    env = get_env("whole", "cpu", raiz=tmp_path, ejecutar=git_falso(commit=USUARIO_FICTICIO))

    assert env["commit"] is None


def test_lo_seguro_no_se_toca(tmp_path, git_falso, usuario_ficticio):
    env = get_env("owlvit", "cpu", raiz=tmp_path, ejecutar=git_falso())

    assert env["detector"] == "owlvit" and env["commit"] == "abc1234"
    assert env["python"] == platform.python_version()


def test_caso6_con_las_funciones_reales_ningun_valor_lleva_rutas_ni_usuario():
    env = get_env("whole", "cpu")  # git, platform y metadata reales

    assert list(env) == CLAVES
    for clave, valor in env.items():
        assert _seguro(valor), f"{clave}={valor!r}"
