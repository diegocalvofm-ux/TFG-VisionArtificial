"""Entorno de un entrenamiento y comprobación de que la instalación no cambia torch.

Uso desde el notebook de Colab (o desde PowerShell en local):
    python -m src.train.entorno versiones --out antes.json
    python -m src.train.entorno constraints --out constraints.txt
    pip install -c constraints.txt -r requirements-train.txt
    python -m src.train.entorno versiones --out despues.json
    python -m src.train.entorno comprobar --antes antes.json --despues despues.json

``constraints`` escribe ``torch==<instalado>`` y ``torchvision==<instalado>`` para que pip no pueda
sustituirlos. ``comprobar`` compara las versiones de antes y después de instalar e imprime si
cambiaron (sale con código 1 si cambió alguna). Las versiones se leen de los metadatos de los
paquetes, sin importar torch (importarlo tarda y en Colab fijaría la versión en memoria).
"""

import argparse
import importlib.metadata
import importlib.util
import json
import platform
import subprocess
import sys
from pathlib import Path

from src.utils.env import _git, _sin_datos_personales

RAIZ = Path(__file__).resolve().parents[2]

# Paquetes cuya versión se guarda con cada ejecución (nombre de distribución en PyPI).
PAQUETES_ENTORNO = (
    "torch", "torchvision", "rfdetr", "transformers", "pytorch-lightning", "numpy",
)  # fmt: skip
# Los dos que la instalación no debe tocar.
PAQUETES_TORCH = ("torch", "torchvision")


def versiones_paquetes(nombres=PAQUETES_ENTORNO) -> dict[str, str | None]:
    """Versión instalada de cada paquete (``None`` si no está), leída de sus metadatos."""
    salida = {}
    for nombre in nombres:
        try:
            salida[nombre] = importlib.metadata.version(nombre)
        except importlib.metadata.PackageNotFoundError:
            salida[nombre] = None
    return salida


def _info_gpu() -> dict:
    """Nombre y memoria total de la GPU 0, o ``None`` si no hay torch o no hay GPU."""
    vacio = {"nombre": None, "memoria_total_mb": None}
    try:
        import torch
    except ImportError:
        return vacio
    if not torch.cuda.is_available():
        return vacio
    propiedades = torch.cuda.get_device_properties(0)
    return {
        "nombre": torch.cuda.get_device_name(0),
        "memoria_total_mb": round(propiedades.total_memory / 1024**2),
    }


def _plataforma() -> str:
    """``colab`` si se ejecuta en Google Colab y ``local`` en cualquier otro sitio."""
    try:
        return "colab" if importlib.util.find_spec("google.colab") else "local"
    except (ImportError, ValueError):
        return "local"


def entorno_entrenamiento(
    *, raiz: Path = RAIZ, ejecutar=subprocess.run, versiones=versiones_paquetes, gpu=_info_gpu,
    plataforma=None,
) -> dict:  # fmt: skip
    """Describe el entorno del entrenamiento en un diccionario listo para guardar como JSON.

    Claves: ``python``, ``torch``, ``torchvision``, ``rfdetr``, ``transformers``,
    ``pytorch_lightning``, ``numpy`` (versiones; ``None`` si no están), ``gpu`` y
    ``gpu_memoria_mb`` (``None`` sin GPU), ``sistema``, ``plataforma`` (``colab`` o ``local``),
    ``commit`` (``git rev-parse --short HEAD``) y ``dirty`` (``True`` si ``src`` o ``configs``
    tienen cambios sin commitear). Sin git, ``commit`` y ``dirty`` son ``None``. Un valor de
    texto con una ruta absoluta o con el usuario del sistema se sustituye por ``None``: el
    repositorio es público. ``ejecutar``, ``versiones``, ``gpu`` y ``plataforma`` se pueden
    sustituir al probar.
    """
    paquetes = versiones()
    info_gpu = gpu()
    commit = _git(raiz, ["rev-parse", "--short", "HEAD"], ejecutar)
    estado = _git(raiz, ["status", "--porcelain", "--", "src", "configs"], ejecutar)
    env = {
        "python": platform.python_version(),
        "torch": paquetes.get("torch"),
        "torchvision": paquetes.get("torchvision"),
        "rfdetr": paquetes.get("rfdetr"),
        "transformers": paquetes.get("transformers"),
        "pytorch_lightning": paquetes.get("pytorch-lightning"),
        "numpy": paquetes.get("numpy"),
        "gpu": info_gpu["nombre"],
        "gpu_memoria_mb": info_gpu["memoria_total_mb"],
        "sistema": f"{platform.system()} {platform.release()}".strip() or None,
        "plataforma": plataforma or _plataforma(),
        "commit": (commit or "").strip() or None,
        "dirty": None if estado is None else bool(estado.strip()),
    }
    return {clave: _sin_datos_personales(valor) for clave, valor in env.items()}


def comprobar_torch(antes: dict, despues: dict) -> tuple[bool, list[str]]:
    """Compara las versiones de torch y torchvision de antes y después de instalar.

    Devuelve ``(sin_cambios, lineas)``: una línea por paquete, p. ej.
    ``torch: 2.11.0 -> 2.11.0 (sin cambios)`` o ``... (CAMBIÓ)``. Que un paquete aparezca o
    desaparezca también cuenta como cambio.
    """
    sin_cambios, lineas = True, []
    for paquete in PAQUETES_TORCH:
        a, d = antes.get(paquete), despues.get(paquete)
        igual = a == d
        sin_cambios = sin_cambios and igual
        lineas.append(f"{paquete}: {a} -> {d} ({'sin cambios' if igual else 'CAMBIÓ'})")
    return sin_cambios, lineas


def crear_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.train.entorno",
        description="Versiones de torch y comprobación de que la instalación no las cambia.",
    )
    ordenes = parser.add_subparsers(dest="orden", required=True)
    versiones_ = ordenes.add_parser("versiones", help="guarda la versión de torch y torchvision")
    versiones_.add_argument("--out", required=True, type=Path, help="JSON de salida")
    constraints = ordenes.add_parser("constraints", help="escribe un constraints que fija torch")
    constraints.add_argument("--out", required=True, type=Path, help="archivo para pip -c")
    comprobar = ordenes.add_parser("comprobar", help="compara antes y después de instalar")
    comprobar.add_argument("--antes", required=True, type=Path)
    comprobar.add_argument("--despues", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Ejecuta la orden pedida. Devuelve 0 si todo va bien y 1 si hay un error o torch cambió."""
    args = crear_parser().parse_args(argv)
    if args.orden == "versiones":
        instaladas = versiones_paquetes(PAQUETES_TORCH)
        actuales = {p: instaladas.get(p) for p in PAQUETES_TORCH}
        args.out.write_text(json.dumps(actuales, indent=1), encoding="utf-8")
        print("  ".join(f"{p}={v}" for p, v in actuales.items()))
        return 0
    if args.orden == "constraints":
        actuales = versiones_paquetes(PAQUETES_TORCH)
        if actuales.get("torch") is None:
            print("ERROR: torch no está instalado: no hay versión que fijar", file=sys.stderr)
            return 1
        lineas = [f"{p}=={actuales[p]}" for p in PAQUETES_TORCH if actuales.get(p)]
        args.out.write_text("\n".join(lineas) + "\n", encoding="utf-8")
        print("Fijado: " + ", ".join(lineas))
        return 0
    antes = json.loads(args.antes.read_text(encoding="utf-8"))
    despues = json.loads(args.despues.read_text(encoding="utf-8"))
    sin_cambios, lineas = comprobar_torch(antes, despues)
    print("\n".join(lineas))
    if not sin_cambios:
        print(
            "ERROR: la instalación cambió torch o torchvision; reinicia el entorno y revisa "
            "requirements-train.txt",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
