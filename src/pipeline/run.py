"""Pipeline de extremo a extremo: carpeta de imágenes -> un JSON y una imagen anotada por foto.

Uso:
    python -m src.pipeline.run --input DIR --out DIR --detector whole

Etapa 1: el detector localiza las lechugas. Etapa 2: el asesor da una etiqueta a cada uno
de los seis grupos. Después ``aggregate`` resume los seis grupos en un nivel global
(ACCEPT / REVIEW / NON-CONFORMING) usando la tabla de ``configs/salida_global.yaml``.
"""

import argparse
import json
import platform
import sys
import time
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path

import cv2
import numpy as np
import yaml

from src.config import load_config
from src.evaluation.aggregate import aggregate
from src.pipeline.assessors.null import NullAssessor
from src.pipeline.detectors import crear_detector, nombres_detectores
from src.pipeline.draw import dibujar
from src.pipeline.imagenes import EXTENSIONES, guardar_imagen, leer_imagen
from src.pipeline.types import Assessor, Detector

# Raíz del repositorio: las rutas por defecto se calculan desde aquí, no desde la carpeta
# en la que se lance el comando, para que no aparezca un "runs/" en un sitio inesperado.
RAIZ = Path(__file__).resolve().parents[2]


def crear_parser() -> argparse.ArgumentParser:
    """Define los argumentos de la línea de comandos."""
    parser = argparse.ArgumentParser(
        prog="python -m src.pipeline.run",
        description="Localiza las lechugas de una carpeta de imágenes y evalúa su calidad.",
    )
    parser.add_argument("--input", required=True, type=Path, help="carpeta con las imágenes")
    parser.add_argument(
        "--out", type=Path, default=RAIZ / "runs" / "v0", help="carpeta de salida (runs/v0)"
    )
    parser.add_argument(
        "--detector",
        default=None,
        help=(
            f"detector a usar (disponibles: {', '.join(nombres_detectores())}); "
            "por defecto, el de la configuración"
        ),
    )
    parser.add_argument(
        "--config", type=Path, default=RAIZ / "configs" / "v0.yaml", help="configuración (YAML)"
    )
    parser.add_argument(
        "--tabla",
        type=Path,
        default=RAIZ / "configs" / "salida_global.yaml",
        help="tabla de salida global (YAML)",
    )
    return parser


def procesar_imagen(
    imagen: np.ndarray, nombre: str, detector: Detector, assessor: Assessor, tabla: Mapping
) -> dict:
    """Procesa una imagen ya cargada y devuelve el resultado listo para guardar como JSON.

    En lenguaje llano: pide al detector las lechugas; para cada una pide al asesor la
    etiqueta de los seis grupos y calcula el nivel global con ``aggregate``. Mide cuánto
    tarda cada etapa (en milisegundos, con ``time.perf_counter``). El ``total`` cubre la
    detección y la evaluación, pero no leer ni escribir archivos. Si no hay detecciones,
    ``lettuces`` queda como lista vacía.
    """
    inicio = time.perf_counter()
    antes = time.perf_counter()
    detecciones = detector.detect(imagen)
    ms_detectar = (time.perf_counter() - antes) * 1000

    lechugas = []
    ms_evaluar = 0.0
    for deteccion in detecciones:
        antes = time.perf_counter()
        grupos = assessor.assess(imagen, deteccion)
        ms_evaluar += (time.perf_counter() - antes) * 1000
        agregado = aggregate(grupos, tabla)
        lechugas.append(
            {
                "box": [int(valor) for valor in deteccion.box],
                "score": float(deteccion.score),
                "groups": dict(grupos),
                "global": agregado.level,
                "provisional_labels": agregado.provisional_labels,
            }
        )
    ms_total = (time.perf_counter() - inicio) * 1000

    return {
        "image": nombre,
        "lettuces": lechugas,
        "timing_ms": {"detect": ms_detectar, "assess": ms_evaluar, "total": ms_total},
        "env": {"python": platform.python_version(), "opencv": cv2.__version__},
    }


def _escribir_json(ruta: Path, datos: dict) -> None:
    """Escribe el JSON en UTF-8 y sin escapar la ñ (hay claves como ``daño_fisico``)."""
    with ruta.open("w", encoding="utf-8", newline="\n") as archivo:
        json.dump(datos, archivo, ensure_ascii=False, indent=2)
        archivo.write("\n")


def procesar_carpeta(
    entrada: Path, salida: Path, detector: Detector, assessor: Assessor, tabla: Mapping
) -> tuple[int, int]:
    """Procesa todas las imágenes de una carpeta y devuelve ``(procesadas, saltadas)``.

    En lenguaje llano: recorre los archivos de la carpeta por orden alfabético (sin entrar
    en subcarpetas). Por cada imagen escribe ``<nombre>.json`` y ``<nombre>_anotada.jpg`` en
    la carpeta de salida. Lo que no se puede procesar (extensión no admitida, imagen
    corrupta, o un nombre repetido con otra extensión) se salta con un aviso, y el
    proceso sigue con el resto.
    """
    entrada, salida = Path(entrada), Path(salida)
    salida.mkdir(parents=True, exist_ok=True)
    procesadas = saltadas = 0
    ya_usados: dict[str, str] = {}  # nombre sin extensión -> archivo que lo ocupó

    for ruta in sorted(entrada.iterdir(), key=lambda p: p.name):
        if ruta.is_dir():
            continue
        if ruta.suffix.lower() not in EXTENSIONES:
            warnings.warn(
                f"Se salta '{ruta.name}': extensión no admitida "
                f"(admitidas: {', '.join(sorted(EXTENSIONES))})",
                UserWarning,
                stacklevel=2,
            )
            saltadas += 1
            continue
        clave = ruta.stem.casefold()
        if clave in ya_usados:
            warnings.warn(
                f"Se salta '{ruta.name}': tiene el mismo nombre que '{ya_usados[clave]}' y "
                "sus resultados se sobrescribirían",
                UserWarning,
                stacklevel=2,
            )
            saltadas += 1
            continue
        try:
            imagen = leer_imagen(ruta)
        except (OSError, ValueError) as error:
            warnings.warn(
                f"Se salta '{ruta.name}': no se puede leer la imagen ({error})",
                UserWarning,
                stacklevel=2,
            )
            saltadas += 1
            continue

        ya_usados[clave] = ruta.name
        resultado = procesar_imagen(imagen, ruta.name, detector, assessor, tabla)
        _escribir_json(salida / f"{ruta.stem}.json", resultado)
        guardar_imagen(salida / f"{ruta.stem}_anotada.jpg", dibujar(imagen, resultado["lettuces"]))
        procesadas += 1

    return procesadas, saltadas


def main(argv: Sequence[str] | None = None) -> int:
    """Punto de entrada de la línea de comandos. Devuelve el código de salida.

    ``0``: se procesó al menos una imagen. ``1``: no se procesó ninguna. ``2``: error de
    uso (carpeta o configuración inexistentes, detector desconocido...), con el motivo por
    la salida de errores. Recibe ``argv`` para poder probarlo sin lanzar otro proceso.
    """
    argumentos = crear_parser().parse_args(argv)

    if not argumentos.input.is_dir():
        print(f"Error: la carpeta de entrada no existe: {argumentos.input}", file=sys.stderr)
        return 2
    try:
        cfg = load_config(argumentos.config)
        tabla = load_config(argumentos.tabla)
        nombre = argumentos.detector or cfg.get("detector")
        if not nombre:
            raise ValueError("no se indicó detector ni hay 'detector' en la configuración")
        detector = crear_detector(nombre, cfg)
    except (OSError, ValueError, yaml.YAMLError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2

    procesadas, saltadas = procesar_carpeta(
        argumentos.input, argumentos.out, detector, NullAssessor(), tabla
    )
    print(f"Procesadas: {procesadas} | Saltadas: {saltadas} | Salida: {argumentos.out}")
    if procesadas == 0:
        print("Error: no se procesó ninguna imagen", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
