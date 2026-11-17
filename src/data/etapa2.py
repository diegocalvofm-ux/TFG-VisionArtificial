"""Línea de comandos del dataset de recortes de la etapa 2.

Uso (desde la raíz del repositorio, en PowerShell):
    python -m src.data.etapa2 --config configs/datasets/etapa2_v1.yaml --comprobar
    python -m src.data.etapa2 --config configs/datasets/etapa2_v1.yaml

Con ``--comprobar`` valida el YAML y lee las anotaciones y metadata.csv para mostrar cuántos
recortes saldrían de cada fuente y partición, sin cargar RF-DETR ni escribir nada (es rápido).
Sin él, escribe ``imagenes/``, ``control/``, ``recortes.csv`` e ``informe.json`` en la carpeta
``salida`` del YAML, sustituyendo la anterior. El avance se muestra por la salida de errores. La
lógica está en ``src/data/dataset_etapa2.py``.
"""

import argparse
import sys
import zipfile
from pathlib import Path

from src.config import load_config
from src.data.dataset_etapa2 import construir_dataset, formatear_informe
from src.experimentos.privacidad import sanear

# Raíz del repositorio: de aquí salen data/, configs/ y runs/ por defecto.
RAIZ = Path(__file__).resolve().parents[2]


def crear_parser() -> argparse.ArgumentParser:
    """Define los argumentos de la línea de comandos."""
    parser = argparse.ArgumentParser(
        prog="python -m src.data.etapa2",
        description="Construye el dataset de recortes de la etapa 2 (propio, YSSP y PHS).",
    )
    parser.add_argument(
        "--config", required=True, type=Path, help="p. ej. configs/datasets/etapa2_v1.yaml"
    )
    parser.add_argument(
        "--raiz", type=Path, default=RAIZ, help="raíz a la que son relativas las rutas del YAML"
    )
    parser.add_argument(
        "--tabla",
        type=Path,
        default=RAIZ / "configs" / "salida_global.yaml",
        help="tabla de salida global (por defecto, configs/salida_global.yaml)",
    )
    parser.add_argument(
        "--comprobar",
        action="store_true",
        help="cuenta los candidatos sin cargar RF-DETR ni escribir nada",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Construye el dataset. Devuelve 0 si todo va bien y 1 si hay un error (con mensaje)."""
    # La consola de Windows (cp1252) no sabe escribir algunos caracteres: se sustituyen en lugar
    # de hacer fallar el programa.
    for flujo in (sys.stdout, sys.stderr):
        if hasattr(flujo, "reconfigure"):
            flujo.reconfigure(errors="replace")

    argumentos = crear_parser().parse_args(argv)

    def progreso(texto: str) -> None:
        print(texto, file=sys.stderr, flush=True)

    try:
        informe = construir_dataset(
            argumentos.config,
            raiz=argumentos.raiz,
            tabla=load_config(argumentos.tabla),
            comprobar=argumentos.comprobar,
            progreso=progreso,
        )
    except (ValueError, OSError, ImportError, zipfile.BadZipFile) as error:
        # Los errores del sistema operativo traen la ruta completa: se dejan relativas.
        print(f"ERROR: {sanear(str(error), argumentos.raiz)}", file=sys.stderr)
        return 1
    print(formatear_informe(informe))
    if argumentos.comprobar:
        print("(--comprobar: no se ha escrito nada)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
