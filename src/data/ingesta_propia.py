"""Línea de comandos de la ingesta de fotos propias.

Uso (desde la raíz del repositorio, en PowerShell):
    python -m src.data.ingesta_propia --config configs/datasets/propio_piloto.yaml --comprobar
    python -m src.data.ingesta_propia --config configs/datasets/propio_piloto.yaml

Con ``--comprobar`` lee y valida todo (fotos, hoja, reglas, metadata.csv fusionado) y muestra el
resumen, pero no escribe nada: sirve para revisar la hoja antes de la primera ingesta. Sin él,
escribe las fotos limpias e ``informe.json`` en la carpeta ``salida`` del YAML y sus filas en
``data/metadata.csv``. La lógica está en ``src/data/propio.py``.
"""

import argparse
import sys
from pathlib import Path

from src.config import load_config
from src.data.propio import formatear_informe, ingerir

# Raíz del repositorio: de aquí salen data/ y configs/ por defecto.
RAIZ = Path(__file__).resolve().parents[2]


def crear_parser() -> argparse.ArgumentParser:
    """Define los argumentos de la línea de comandos."""
    parser = argparse.ArgumentParser(
        prog="python -m src.data.ingesta_propia",
        description="Ingiere una sesión de fotos propias: fotos limpias, metadata.csv e informe.",
    )
    parser.add_argument(
        "--config", required=True, type=Path, help="p. ej. configs/datasets/propio_piloto.yaml"
    )
    parser.add_argument(
        "--raiz", type=Path, default=RAIZ, help="raíz a la que son relativas las rutas del YAML"
    )
    parser.add_argument(
        "--metadata", type=Path, default=None, help="por defecto, <raiz>/data/metadata.csv"
    )
    parser.add_argument(
        "--tabla",
        type=Path,
        default=RAIZ / "configs" / "salida_global.yaml",
        help="tabla de salida global (por defecto, configs/salida_global.yaml)",
    )
    parser.add_argument(
        "--comprobar", action="store_true", help="valida y muestra el resumen sin escribir nada"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Ejecuta la ingesta. Devuelve 0 si todo va bien y 1 si hay un error (con mensaje)."""
    # La consola de Windows (cp1252) no sabe escribir algunos caracteres: se sustituyen en lugar
    # de hacer fallar el programa.
    for flujo in (sys.stdout, sys.stderr):
        if hasattr(flujo, "reconfigure"):
            flujo.reconfigure(errors="replace")

    argumentos = crear_parser().parse_args(argv)
    try:
        informe = ingerir(
            argumentos.config,
            raiz=argumentos.raiz,
            metadata=argumentos.metadata or argumentos.raiz / "data" / "metadata.csv",
            tabla=load_config(argumentos.tabla),
            comprobar=argumentos.comprobar,
        )
    except (ValueError, OSError, ImportError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(formatear_informe(informe))
    if argumentos.comprobar:
        print("(--comprobar: no se ha escrito nada)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
