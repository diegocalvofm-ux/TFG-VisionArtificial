"""Privacidad: que ninguna nota ni mensaje del script lleve rutas absolutas ni el usuario.

El repositorio es público y las notas de experimentos se versionan. Una ruta como
``C:\\Users\\<nombre>\\...`` revela el nombre del usuario del sistema; por eso las rutas se
guardan siempre relativas a la raíz del repositorio y todo lo que el script imprime pasa por
``sanear``.
"""

import getpass
import re
from pathlib import Path

# Un nombre de usuario de menos de 3 letras daría falsos positivos con cualquier palabra corta.
_USUARIO_MINIMO = 3

# Rutas de Windows (C:\x o C:/x, también con barras dobles como las muestra un OSError) y
# rutas de red (\\servidor\recurso).
_RUTA_WINDOWS = re.compile(r"(?<!\w)(?:[A-Za-z]:[\\/]|\\\\[^\s\\/'\"<>|]+[\\/])[^\s'\"<>|]*")

# Rutas de Linux, macOS y Colab: empiezan por "/" y tienen al menos dos tramos
# (/home/x, /content/drive...). Lo que va pegado a una palabra (a/b, 1/2/3) o a "://" no cuenta.
_RUTA_POSIX = re.compile(r"(?<![\w/.:~\\-])/[^\s/'\"<>|]+(?:/[^\s'\"<>|]*)+")


def contiene_ruta_absoluta(texto: str) -> bool:
    """Indica si el texto contiene una ruta absoluta de Windows, Linux o macOS."""
    return bool(_RUTA_WINDOWS.search(texto) or _RUTA_POSIX.search(texto))


def _patron_usuario() -> re.Pattern | None:
    """Patrón que reconoce el nombre de usuario del sistema como palabra completa."""
    try:
        nombre = getpass.getuser()
    except (OSError, KeyError, ImportError):  # sin usuario identificable
        return None
    if len(nombre) < _USUARIO_MINIMO:
        return None
    return re.compile(rf"(?<!\w){re.escape(nombre)}(?!\w)", re.IGNORECASE)


def contiene_usuario(texto: str) -> bool:
    """Indica si el texto contiene el nombre de usuario del sistema (sin distinguir mayúsculas)."""
    patron = _patron_usuario()
    return bool(patron and patron.search(texto))


def validar(clave: str, valor: str) -> None:
    """Lanza ValueError si un valor que se va a escribir en una nota revela datos privados.

    En lenguaje llano: se rechaza (en vez de reescribirlo en silencio) cualquier texto con una
    ruta absoluta o con el nombre de usuario, para que quien lo escribe lo corrija. El mensaje
    no repite el valor, así no vuelve a imprimirse lo que se quería ocultar.
    """
    if not isinstance(valor, str):
        return
    if contiene_ruta_absoluta(valor):
        raise ValueError(
            f"El valor de '{clave}' contiene una ruta absoluta; usa una ruta relativa al "
            "repositorio"
        )
    if contiene_usuario(valor):
        raise ValueError(f"El valor de '{clave}' contiene el nombre de usuario del sistema")


def _variantes(base: Path) -> list[str]:
    """Formas en que puede aparecer escrita la raíz en un mensaje, de la más larga a la corta."""
    escritas = {str(base), base.as_posix()}
    try:
        resuelta = base.resolve()
        escritas |= {str(resuelta), resuelta.as_posix()}
    except OSError:
        pass
    escritas |= {e.replace("\\", "\\\\") for e in escritas}  # como la muestra un OSError
    return sorted((e for e in escritas if e), key=len, reverse=True)


def sanear(texto: str, base: Path | None = None) -> str:
    """Quita de un mensaje las rutas absolutas y el nombre de usuario.

    En lenguaje llano, en este orden: 1) lo que está dentro de la raíz del repositorio ``base``
    pasa a ser relativo (``C:\\repo\\docs\\x`` queda ``docs\\x``); 2) cualquier otra ruta absoluta
    se sustituye por ``<ruta>``; 3) el nombre de usuario del sistema por ``<usuario>``. Se usa
    como red de seguridad con todo lo que se imprime, incluidos los textos de error del sistema
    operativo, que llevan la ruta completa. Aplicarlo dos veces da lo mismo que una.
    """
    if base is not None:
        for variante in _variantes(Path(base)):
            patron = re.escape(variante) + r"(?:([\\/]+)|(?![\w.\-]))"
            texto = re.sub(patron, lambda m: "" if m.group(1) else ".", texto)
    texto = _RUTA_WINDOWS.sub("<ruta>", texto)
    texto = _RUTA_POSIX.sub("<ruta>", texto)
    patron_usuario = _patron_usuario()
    return patron_usuario.sub("<usuario>", texto) if patron_usuario else texto


def relativa(ruta: Path, base: Path) -> str:
    """Ruta relativa a ``base`` y con barras normales (``configs/v0.yaml``).

    Error (ValueError): la ruta está fuera de ``base``; no se guarda una ruta externa.
    """
    try:
        relativa_ = Path(ruta).resolve().relative_to(Path(base).resolve())
    except ValueError:
        raise ValueError(
            "La ruta debe estar dentro del repositorio: usa un archivo o una carpeta del proyecto"
        ) from None
    return relativa_.as_posix()


def mostrar(ruta: Path, base: Path) -> str:
    """Ruta para enseñar al usuario: relativa a ``base`` o, si está fuera, solo su nombre."""
    try:
        return relativa(ruta, base)
    except ValueError:
        return Path(ruta).name
