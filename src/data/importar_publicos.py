"""Línea de comandos del importador de datasets públicos (COCO de Roboflow).

Uso (desde la raíz del repositorio, en PowerShell):
    python -m src.data.importar_publicos listar --zip data/Raw/<archivo>.zip
    python -m src.data.importar_publicos importar --etapa configs/datasets/etapa1_v1.yaml
    python -m src.data.importar_publicos revisar --salida data/processed/etapa1_v1 --particion valid

``listar`` solo lee el ZIP y no escribe nada. ``importar`` genera ``data/metadata.csv`` y el
dataset ``data/processed/etapa1_v1`` (borrando antes la salida anterior). ``revisar`` dibuja
imágenes de ejemplo en ``runs/checks/`` para comprobarlas a ojo. La lógica está en
``src/data/publicos.py``.
"""

import argparse
import sys
from pathlib import Path

from src.config import load_config
from src.data.publicos import (
    PARTICIONES_EXPORTADAS,
    describir_zip,
    formatear_informe,
    importar,
    revisar,
)

# Raíz del repositorio: de aquí salen data/, configs/ y runs/ por defecto.
RAIZ = Path(__file__).resolve().parents[2]


def crear_parser() -> argparse.ArgumentParser:
    """Define las órdenes y los argumentos de la línea de comandos."""
    parser = argparse.ArgumentParser(
        prog="python -m src.data.importar_publicos",
        description="Importa datasets públicos COCO (Roboflow) a metadata.csv y a la etapa 1.",
    )
    ordenes = parser.add_subparsers(dest="orden", required=True)

    listar = ordenes.add_parser("listar", help="muestra clases, imágenes y nombres de un ZIP")
    listar.add_argument("--zip", required=True, type=Path, help="ZIP de Roboflow (solo se lee)")

    imp = ordenes.add_parser("importar", help="genera metadata.csv y el dataset de la etapa 1")
    imp.add_argument(
        "--etapa", required=True, type=Path, help="p. ej. configs/datasets/etapa1_v1.yaml"
    )
    imp.add_argument(
        "--raiz", type=Path, default=RAIZ, help="raíz a la que son relativas las rutas"
    )
    imp.add_argument(
        "--metadata", type=Path, default=None, help="por defecto, <raiz>/data/metadata.csv"
    )
    imp.add_argument(
        "--tabla",
        type=Path,
        default=RAIZ / "configs" / "salida_global.yaml",
        help="tabla de salida global (por defecto, configs/salida_global.yaml)",
    )
    imp.add_argument("--semilla", type=int, default=None, help="sustituye a la semilla del YAML")

    rev = ordenes.add_parser("revisar", help="dibuja cajas y polígonos de imágenes al azar")
    rev.add_argument("--salida", required=True, type=Path, help="dataset exportado")
    rev.add_argument("--particion", required=True, choices=PARTICIONES_EXPORTADAS)
    rev.add_argument("--n", type=int, default=12, help="cuántas imágenes dibujar (12 por defecto)")
    rev.add_argument("--semilla", type=int, default=0)
    rev.add_argument(
        "--destino", type=Path, default=RAIZ / "runs" / "checks", help="por defecto, runs/checks"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Ejecuta la orden pedida. Devuelve 0 si todo va bien y 1 si hay un error (con mensaje)."""
    # Los nombres de archivo pueden llevar caracteres que la consola de Windows (cp1252) no
    # sabe escribir: se sustituyen en lugar de hacer fallar el programa.
    for flujo in (sys.stdout, sys.stderr):
        if hasattr(flujo, "reconfigure"):
            flujo.reconfigure(errors="replace")

    argumentos = crear_parser().parse_args(argv)
    try:
        if argumentos.orden == "listar":
            print(describir_zip(argumentos.zip))
        elif argumentos.orden == "importar":
            metadata = argumentos.metadata or argumentos.raiz / "data" / "metadata.csv"
            informe = importar(
                argumentos.etapa,
                raiz=argumentos.raiz,
                metadata=metadata,
                tabla=load_config(argumentos.tabla),
                semilla=argumentos.semilla,
            )
            print(formatear_informe(informe))
        else:
            escritas = revisar(
                argumentos.salida,
                argumentos.particion,
                argumentos.n,
                argumentos.semilla,
                argumentos.destino,
            )
            print(
                f"{len(escritas)} imágenes dibujadas en {argumentos.destino / argumentos.particion}"
            )
    except (ValueError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
