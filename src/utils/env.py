"""Entorno de una ejecución: versiones, dispositivo, y commit y estado de git del código.

Se guarda en el JSON de cada imagen para poder reproducir y comparar ejecuciones (local frente
a Colab, CPU frente a GPU). Ningún valor lleva rutas absolutas ni el nombre de usuario del
sistema: el repositorio es público y esos datos no deben acabar en las notas ni en los JSON.
"""

import getpass
import importlib.metadata
import platform
import re
import subprocess
from pathlib import Path

import cv2

# Raíz del repositorio: ahí se ejecuta git.
RAIZ = Path(__file__).resolve().parents[2]

_DISPOSITIVOS = ("cpu", "cuda")

# Rutas absolutas de Windows (C:\x, C:/x, \\servidor\x) y de Linux/macOS (/home/x/y).
_RUTA_WINDOWS = re.compile(r"(?<!\w)(?:[A-Za-z]:[\\/]|\\\\[^\s\\/'\"<>|]+[\\/])[^\s'\"<>|]*")
_RUTA_POSIX = re.compile(r"(?<![\w/.:~\\-])/[^\s/'\"<>|]+(?:/[^\s'\"<>|]*)+")


def _sin_datos_personales(valor):
    """Devuelve ``None`` si un texto lleva una ruta absoluta o el nombre de usuario del sistema.

    En lenguaje llano: es la red de seguridad de la privacidad. Si algún dato del entorno
    (la versión del sistema, el nombre de un detector...) trajera un dato personal, se
    descarta en lugar de guardarlo. Lo que no es texto (booleanos, ``None``) pasa tal cual.
    """
    if not isinstance(valor, str):
        return valor
    if _RUTA_WINDOWS.search(valor) or _RUTA_POSIX.search(valor):
        return None
    try:
        usuario = getpass.getuser()
    except (OSError, KeyError, ImportError):  # sin usuario identificable
        usuario = ""
    if len(usuario) >= 3 and re.search(rf"(?<!\w){re.escape(usuario)}(?!\w)", valor, re.IGNORECASE):
        return None
    return valor


def _version_torch() -> str | None:
    """Versión de PyTorch instalada, o ``None`` si no está.

    Se lee de los metadatos del paquete en lugar de importar ``torch`` (importarlo tarda
    segundos y con el detector ``whole`` no hace falta). Coincide con ``torch.__version__``.
    """
    try:
        return importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        return None


def _git(raiz: Path, argumentos: list[str], ejecutar) -> str | None:
    """Salida de un comando de git en ``raiz``, o ``None`` si falla o git no está instalado."""
    try:
        proceso = ejecutar(
            ["git", "-C", str(raiz), *argumentos],
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )
    except OSError:  # incluye FileNotFoundError: git no está instalado
        return None
    return proceso.stdout if proceso.returncode == 0 else None


def get_env(detector: str, dispositivo: str, *, raiz: Path = RAIZ, ejecutar=subprocess.run) -> dict:
    """Describe el entorno de la ejecución en un diccionario listo para guardar como JSON.

    Claves, en este orden: ``python`` y ``opencv`` (versiones), ``torch`` (versión o ``None`` si
    no está instalado), ``sistema`` (sistema operativo y versión, sin nombre de equipo ni
    usuario), ``device`` (``"cpu"`` o ``"cuda"``), ``detector``, ``commit`` (``git rev-parse
    --short HEAD`` en la raíz del repo) y ``dirty`` (``True`` si ``git status --porcelain --
    src configs`` devuelve algo, es decir, si el código tiene cambios sin commitear).

    Si no hay git o la carpeta no es un repositorio, ``commit`` y ``dirty`` son ``None``, sin
    error. Un valor de texto con una ruta absoluta o con el usuario del sistema se sustituye por
    ``None``.

    ``dispositivo`` debe ser el dispositivo ya decidido (``cpu`` o ``cuda``, no ``auto``).
    ``raiz`` y ``ejecutar`` (el ``subprocess.run`` con el que se llama a git) se pueden
    sustituir para probarlo.

    Error: ``ValueError`` si ``dispositivo`` no es ``cpu`` ni ``cuda``.
    """
    if dispositivo not in _DISPOSITIVOS:
        raise ValueError(f"device debe ser 'cpu' o 'cuda' (recibido: {dispositivo!r})")

    commit = _git(raiz, ["rev-parse", "--short", "HEAD"], ejecutar)
    estado = _git(raiz, ["status", "--porcelain", "--", "src", "configs"], ejecutar)
    env = {
        "python": platform.python_version(),
        "opencv": cv2.__version__,
        "torch": _version_torch(),
        "sistema": f"{platform.system()} {platform.release()}".strip() or None,
        "device": dispositivo,
        "detector": detector,
        "commit": (commit or "").strip() or None,
        "dirty": None if estado is None else bool(estado.strip()),
    }
    return {clave: _sin_datos_personales(valor) for clave, valor in env.items()}
