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
from src.utils.env import get_env
from src.utils.profiling import mediana, medir, rss_mb, validar_repeticiones

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
    parser.add_argument(
        "--umbral",
        type=float,
        default=None,
        help="solo owlvit: umbral de puntuación; sustituye al de la configuración",
    )
    parser.add_argument(
        "--consulta",
        default=None,
        help="solo owlvit: texto de búsqueda; sustituye al de la configuración",
    )
    parser.add_argument(
        "--repeticiones",
        type=int,
        default=5,
        help=(
            "veces que se mide cada imagen; los tiempos del JSON son la mediana (por defecto 5). "
            "Con owlvit cada imagen se infiere repeticiones + calentamiento veces"
        ),
    )
    parser.add_argument(
        "--calentamiento",
        type=int,
        default=1,
        help="pasadas previas que no se miden, para no contar el arranque en frío (por defecto 1)",
    )
    return parser


def procesar_imagen(
    imagen: np.ndarray,
    nombre: str,
    detector: Detector,
    assessor: Assessor,
    tabla: Mapping,
    *,
    env: Mapping | None = None,
    repeticiones: int = 1,
    calentamiento: int = 0,
    reloj=time.perf_counter,
) -> dict:
    """Procesa una imagen ya cargada y devuelve el resultado listo para guardar como JSON.

    En lenguaje llano: pide al detector las lechugas; para cada una pide al asesor la
    etiqueta de los seis grupos y calcula el nivel global con ``aggregate``. Todo eso es un
    "ciclo", y se repite ``calentamiento`` veces sin medir (para no contar el arranque en
    frío) y ``repeticiones`` veces midiendo. En ``timing_ms`` (milisegundos) ``detect`` y
    ``assess`` son las medianas de la detección y de la evaluación de las pasadas medidas, y
    ``total`` la mediana del ciclo completo (no la suma de las otras dos: con varias
    repeticiones pueden no coincidir). El total no incluye leer ni escribir archivos. Las
    lechugas del JSON son las de la última pasada. Con ``repeticiones=1`` y
    ``calentamiento=0`` es una sola medida, como antes.

    Además del resultado de siempre, el JSON lleva ``image_size`` (``[ancho, alto]``),
    ``model_load_ms`` (cuánto tardó en cargarse el modelo del detector; ``None`` si no carga
    modelo), ``rss_mb`` (memoria residente tras las repeticiones, no el pico) y, en
    ``timing_ms``, ``repeticiones`` y ``calentamiento``. ``env`` es el entorno (ver
    ``src.utils.env.get_env``); si no se da, solo las versiones de Python y OpenCV.
    """
    validar_repeticiones(repeticiones, calentamiento)
    ms_detectar: list[float] = []
    ms_evaluar: list[float] = []
    ultimo: dict = {}

    def ciclo() -> None:
        antes = reloj()
        detecciones = detector.detect(imagen)
        ms_detectar.append((reloj() - antes) * 1000)

        lechugas = []
        evaluacion = 0.0
        for deteccion in detecciones:
            antes = reloj()
            grupos = assessor.assess(imagen, deteccion)
            evaluacion += (reloj() - antes) * 1000
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
        ms_evaluar.append(evaluacion)
        ultimo["lechugas"] = lechugas

    ms_total = medir(ciclo, repeticiones, calentamiento, reloj=reloj)
    # Las primeras ``calentamiento`` pasadas son de calentamiento: no entran en las medianas.
    ms_detectar, ms_evaluar = ms_detectar[calentamiento:], ms_evaluar[calentamiento:]

    carga = getattr(detector, "tiempo_carga_s", None)  # solo los detectores con modelo
    alto, ancho = imagen.shape[:2]
    return {
        "image": nombre,
        "lettuces": ultimo["lechugas"],
        "timing_ms": {
            "detect": mediana(ms_detectar),
            "assess": mediana(ms_evaluar),
            "total": ms_total,
            "repeticiones": repeticiones,
            "calentamiento": calentamiento,
        },
        "env": (
            dict(env)
            if env is not None
            else {"python": platform.python_version(), "opencv": cv2.__version__}
        ),
        "image_size": [int(ancho), int(alto)],
        "model_load_ms": None if carga is None else carga * 1000,
        "rss_mb": rss_mb(),
    }


def _escribir_json(ruta: Path, datos: dict) -> None:
    """Escribe el JSON en UTF-8 y sin escapar la ñ (hay claves como ``daño_fisico``)."""
    with ruta.open("w", encoding="utf-8", newline="\n") as archivo:
        json.dump(datos, archivo, ensure_ascii=False, indent=2)
        archivo.write("\n")


def procesar_carpeta(
    entrada: Path,
    salida: Path,
    detector: Detector,
    assessor: Assessor,
    tabla: Mapping,
    *,
    env: Mapping | None = None,
    repeticiones: int = 1,
    calentamiento: int = 0,
) -> tuple[int, int]:
    """Procesa todas las imágenes de una carpeta y devuelve ``(procesadas, saltadas)``.

    En lenguaje llano: recorre los archivos de la carpeta por orden alfabético (sin entrar
    en subcarpetas). Por cada imagen escribe ``<nombre>.json`` y ``<nombre>_anotada.jpg`` en
    la carpeta de salida. Lo que no se puede procesar (extensión no admitida, imagen
    corrupta, o un nombre repetido con otra extensión) se salta con un aviso, y el
    proceso sigue con el resto. ``env``, ``repeticiones`` y ``calentamiento`` se pasan a
    ``procesar_imagen``; el entorno se calcula una vez por ejecución, no por imagen.
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
        resultado = procesar_imagen(
            imagen, ruta.name, detector, assessor, tabla,
            env=env, repeticiones=repeticiones, calentamiento=calentamiento,
        )  # fmt: skip
        _escribir_json(salida / f"{ruta.stem}.json", resultado)
        guardar_imagen(salida / f"{ruta.stem}_anotada.jpg", dibujar(imagen, resultado["lettuces"]))
        procesadas += 1

    return procesadas, saltadas


def _aplicar_opciones(cfg: Mapping, detector: str, argumentos: argparse.Namespace) -> Mapping:
    """Aplica ``--umbral`` y ``--consulta`` sobre una copia de la configuración.

    En lenguaje llano: sirven para probar otros valores sin editar ``configs/v0.yaml``. La
    configuración original no se toca. Solo valen para ``owlvit``; con otro detector se
    avisa de que se ignoran.
    """
    opciones = {"umbral": argumentos.umbral, "consulta": argumentos.consulta}
    opciones = {clave: valor for clave, valor in opciones.items() if valor is not None}
    if not opciones:
        return cfg
    if detector != "owlvit":
        pedidas = ", ".join(f"--{clave}" for clave in opciones)
        warnings.warn(
            f"Se ignora {pedidas}: solo se aplica al detector 'owlvit' (se usa '{detector}')",
            UserWarning,
            stacklevel=2,
        )
        return cfg
    return {**cfg, "owlvit": {**(cfg.get("owlvit") or {}), **opciones}}


def main(argv: Sequence[str] | None = None) -> int:
    """Punto de entrada de la línea de comandos. Devuelve el código de salida.

    ``0``: se procesó al menos una imagen. ``1``: no se procesó ninguna. ``2``: error de
    uso (carpeta o configuración inexistentes, valores inválidos, detector desconocido,
    ``transformers`` sin instalar, fallo al descargar el modelo...), con el motivo por la
    salida de errores. Recibe ``argv`` para poder probarlo sin lanzar otro proceso.
    """
    argumentos = crear_parser().parse_args(argv)

    if not argumentos.input.is_dir():
        print(f"Error: la carpeta de entrada no existe: {argumentos.input}", file=sys.stderr)
        return 2
    try:
        # Antes de cargar ningún modelo: un valor inválido no debe costar la carga.
        validar_repeticiones(argumentos.repeticiones, argumentos.calentamiento)
        cfg = load_config(argumentos.config)
        tabla = load_config(argumentos.tabla)
        nombre = argumentos.detector or cfg.get("detector")
        if not nombre:
            raise ValueError("no se indicó detector ni hay 'detector' en la configuración")
        cfg = _aplicar_opciones(cfg, nombre, argumentos)
        detector = crear_detector(nombre, cfg)
        # El entorno se calcula una sola vez (son dos llamadas a git): vale para todas las imágenes.
        env = get_env(nombre, getattr(detector, "dispositivo", "cpu"))
    except (OSError, ImportError, ValueError, yaml.YAMLError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2

    tiempo_carga = getattr(detector, "tiempo_carga_s", None)  # solo los detectores con modelo
    if tiempo_carga is not None:
        print(f"Carga del modelo: {tiempo_carga:.2f} s")
    procesadas, saltadas = procesar_carpeta(
        argumentos.input, argumentos.out, detector, NullAssessor(), tabla,
        env=env, repeticiones=argumentos.repeticiones, calentamiento=argumentos.calentamiento,
    )  # fmt: skip
    print(f"Procesadas: {procesadas} | Saltadas: {saltadas} | Salida: {argumentos.out}")
    if procesadas == 0:
        print("Error: no se procesó ninguna imagen", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
