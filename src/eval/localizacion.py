"""Evaluación de la localización de lechugas: OWL-ViT, RF-DETR Nano y una línea base HSV.

Uso (desde la raíz del repositorio):
    python -m src.eval.localizacion --config configs/eval/localizacion.yaml --exp-id mi_evaluacion

Compara los métodos activos sobre un conjunto COCO con cajas de referencia (una clase) y guarda en
``runs/<exp_id>/``: ``por_imagen.csv``, ``resumen.json`` (métricas, latencias, entorno, commit y
estado de git), ``tabla.md`` (lista para la memoria) y ``ejemplos/<método>/`` con imágenes que
muestran las cajas de referencia (verde) y las predichas (rojo). Todo parámetro sale del YAML; ver
``src/eval/config.py``. Solo se crea la carpeta de la ejecución cuando ya se han medido todos los
métodos, así un fallo a medias no deja una ejecución incompleta.
"""

import argparse
import json
import random
import re
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from src.eval.cajas import emparejar
from src.eval.config import ConfigLocalizacion, cargar_config_localizacion
from src.eval.conjunto import ImagenGT, cargar_conjunto_coco
from src.eval.metodos import Metodo, crear_fabricas
from src.eval.metricas import ResultadoImagen, agregar, ap_coco, resumen_latencia
from src.eval.salidas import AVISO_VALID, escribir_csv, generar_tabla_md, guardar_ejemplo
from src.experimentos.privacidad import mostrar, sanear
from src.pipeline.imagenes import leer_imagen
from src.pipeline.types import Detection
from src.train.entorno import PAQUETES_ENTORNO, versiones_paquetes
from src.utils.env import get_env
from src.utils.profiling import medir

RAIZ = Path(__file__).resolve().parents[2]
CONFIG_POR_DEFECTO = RAIZ / "configs" / "eval" / "localizacion.yaml"

_ID_EJECUCION = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
COLUMNAS_CSV = (
    "metodo", "imagen", "n_gt", "n_pred", "tp", "fp", "fn", "precision", "recall", "f1",
    "iou_medio", "sin_deteccion", "latencia_ms",
)  # fmt: skip
_DECIMALES_CSV = 6


@dataclass
class ResultadoMetodo:
    """Lo medido con un método: resultados por imagen y las predicciones de cada imagen.

    ``predicciones_ap`` son todas las que devolvió el método (umbral bajo, para el AP) y
    ``predicciones_operativas`` las que superan el umbral operativo (las que cuentan en TP/FP/FN).
    """

    resultados: list[ResultadoImagen]
    predicciones_ap: list[list[Detection]]
    predicciones_operativas: list[list[Detection]]


def evaluar_metodo(
    metodo: Metodo,
    imagenes: Sequence[ImagenGT],
    cfg: ConfigLocalizacion,
    *,
    leer: Callable[[Path], np.ndarray] = leer_imagen,
    reloj: Callable[[], float] = time.perf_counter,
) -> ResultadoMetodo:
    """Mide un método en todas las imágenes: aciertos y errores, y latencia.

    En lenguaje llano: por cada imagen se lee una vez y se llama al método ``calentamiento`` veces
    sin medir y ``repeticiones`` veces midiendo (``medir`` devuelve la mediana de esas pasadas, que
    es la latencia de la imagen); se queda con las cajas de la última pasada. De ellas, las que
    superan el umbral operativo se emparejan con las de referencia. Sin puntuación de confianza
    (línea base) se ordenan por área, de mayor a menor, para que el resultado sea determinista.

    Error: ``ValueError`` si el tamaño real de una imagen no es el que dice el JSON del conjunto.
    """
    resultado = ResultadoMetodo(resultados=[], predicciones_ap=[], predicciones_operativas=[])
    for imagen_gt in imagenes:
        imagen = leer(imagen_gt.ruta)
        alto, ancho = imagen.shape[:2]
        if (ancho, alto) != (imagen_gt.ancho, imagen_gt.alto):
            raise ValueError(
                f"{imagen_gt.nombre}: el tamaño real ({ancho}x{alto}) no coincide con el del "
                f"conjunto ({imagen_gt.ancho}x{imagen_gt.alto})"
            )
        ultimas: list[Detection] = []

        def ciclo():
            ultimas[:] = metodo.predecir(imagen)

        latencia = medir(ciclo, cfg.repeticiones, cfg.calentamiento, reloj=reloj)

        todas = list(ultimas)
        operativas = [d for d in todas if metodo.umbral is None or d.score > metodo.umbral]
        cajas = np.array([d.box for d in operativas], dtype=float).reshape(-1, 4)
        if metodo.tiene_confianza:
            orden = np.array([d.score for d in operativas], dtype=float)
        else:
            orden = (cajas[:, 2] - cajas[:, 0]) * (cajas[:, 3] - cajas[:, 1])
        emparejamiento = emparejar(cajas, orden, imagen_gt.cajas, cfg.iou_min)

        resultado.predicciones_ap.append(todas)
        resultado.predicciones_operativas.append(operativas)
        resultado.resultados.append(
            ResultadoImagen(
                nombre=imagen_gt.nombre,
                n_gt=len(imagen_gt.cajas),
                n_pred=len(operativas),
                tp=len(emparejamiento.pares),
                fp=len(emparejamiento.fp),
                fn=len(emparejamiento.fn),
                ious=[iou for _, _, iou in emparejamiento.pares],
                latencia_ms=latencia,
            )
        )
    return resultado


def _redondear(valor):
    return None if valor is None else round(valor, _DECIMALES_CSV)


def _filas_csv(nombre: str, resultados: Sequence[ResultadoImagen]) -> list[dict]:
    return [
        {
            "metodo": nombre, "imagen": r.nombre, "n_gt": r.n_gt, "n_pred": r.n_pred,
            "tp": r.tp, "fp": r.fp, "fn": r.fn,
            "precision": _redondear(r.precision), "recall": _redondear(r.recall),
            "f1": _redondear(r.f1), "iou_medio": _redondear(r.iou_medio),
            "sin_deteccion": int(r.sin_deteccion), "latencia_ms": _redondear(r.latencia_ms),
        }
        for r in resultados
    ]  # fmt: skip


def _resumen_metodo(
    metodo: Metodo, medido: ResultadoMetodo, imagenes: Sequence[ImagenGT], cfg: ConfigLocalizacion
) -> dict:
    """Todo lo que se guarda en ``resumen.json`` de un método."""
    resultados = medido.resultados
    if metodo.tiene_confianza:
        ap50, ap50_95 = ap_coco(
            [i.cajas for i in imagenes],
            [
                (
                    np.array([d.box for d in dets], dtype=float).reshape(-1, 4),
                    np.array([d.score for d in dets], dtype=float),
                )
                for dets in medido.predicciones_ap
            ],
            [(i.ancho, i.alto) for i in imagenes],
        )
    else:
        ap50 = ap50_95 = None  # sin puntuación de confianza el AP no aplica
    agregado = agregar(resultados)
    return {
        "umbral": metodo.umbral,
        "umbral_ap": metodo.umbral_ap,
        "parametros": metodo.parametros,
        "metricas": {k: agregado[k] for k in ("tp", "fp", "fn", "precision", "recall", "f1",
                                              "iou_medio", "imagenes")},  # fmt: skip
        "imagenes_sin_deteccion": agregado["imagenes_sin_deteccion"],
        "ap50": ap50,
        "ap50_95": ap50_95,
        "latencia": {
            **resumen_latencia([r.latencia_ms for r in resultados]),
            "repeticiones": cfg.repeticiones,
            "calentamiento": cfg.calentamiento,
        },
        "carga_modelo_s": metodo.tiempo_carga_s,
    }


def _hilos_torch() -> int | None:
    """Hilos que usa PyTorch en CPU (condiciona la latencia), o ``None`` sin torch."""
    try:
        import torch

        return int(torch.get_num_threads())
    except Exception:  # sin torch: es solo un dato informativo
        return None


def entorno_evaluacion(dispositivo: str, raiz: Path = RAIZ) -> dict:
    """Entorno de la ejecución: el de ``get_env`` más versiones de paquetes y hilos de PyTorch."""
    env = get_env("localizacion", dispositivo, raiz=raiz)
    env["paquetes"] = versiones_paquetes((*PAQUETES_ENTORNO, "pycocotools"))
    env["hilos_torch"] = _hilos_torch()
    return env


def elegir_ejemplos(total: int, cuantos: int, semilla: int) -> list[int]:
    """Índices de las imágenes de ejemplo: los mismos para todos los métodos (misma semilla)."""
    return sorted(random.Random(semilla).sample(range(total), min(cuantos, total)))


def _escribir_json(ruta: Path, datos: dict) -> None:
    """JSON en UTF-8 y sin escapar la ñ."""
    with ruta.open("w", encoding="utf-8", newline="\n") as archivo:
        json.dump(datos, archivo, ensure_ascii=False, indent=1)
        archivo.write("\n")


def ejecutar(
    config: ConfigLocalizacion,
    exp_id: str,
    *,
    raiz: Path = RAIZ,
    fabricas: dict[str, Callable[[], Metodo]] | None = None,
    entorno: Callable[[], dict] | None = None,
    reloj: Callable[[], float] = time.perf_counter,
) -> dict:
    """Evalúa los métodos activos y guarda todo en ``<salida>/<exp_id>/``. Devuelve el resumen.

    En lenguaje llano, en orden: comprueba el ``exp_id`` y que la ejecución no exista ya; lee el
    conjunto; por cada método activo lo crea (carga el modelo), lo mide en todas las imágenes y
    calcula sus métricas; y solo al final crea la carpeta y escribe el CSV, el resumen, la tabla y
    las imágenes de ejemplo. ``fabricas``, ``entorno`` y ``reloj`` se pueden sustituir al probar.

    Errores: ``ValueError`` (``exp_id`` no válido, ejecución ya existente, tamaño de imagen
    incorrecto), ``FileNotFoundError`` (conjunto o pesos que faltan) e ``ImportError``
    (librería que falta).
    """
    if not _ID_EJECUCION.fullmatch(exp_id or ""):
        raise ValueError(
            f"exp-id no válido: {exp_id!r} (solo letras, números, '-', '_' y '.', sin espacios "
            "ni barras, y sin empezar por un punto)"
        )
    carpeta = raiz / config.salida / exp_id
    if carpeta.exists() and any(carpeta.iterdir()):
        raise ValueError(
            f"La ejecución '{exp_id}' ya existe en {config.salida}/: usa otro --exp-id "
            "(no se pisa una ejecución anterior)"
        )
    fabricas = fabricas or crear_fabricas(config, raiz)
    faltan = [m for m in config.metodos if m not in fabricas]
    if faltan:
        raise ValueError(f"No hay forma de crear los métodos: {', '.join(faltan)}")

    imagenes = cargar_conjunto_coco(raiz / config.conjunto_carpeta / config.particion, config.clase)
    if not imagenes:
        raise ValueError("El conjunto no tiene ninguna imagen")
    env = (entorno or (lambda: entorno_evaluacion(config.dispositivo, raiz)))()
    ejemplos = elegir_ejemplos(len(imagenes), config.ejemplos_numero, config.ejemplos_semilla)

    filas: list[dict] = []
    metodos: dict[str, dict] = {}
    para_ejemplos: dict[str, list[list[Detection]]] = {}
    for nombre in config.metodos:
        metodo = fabricas[nombre]()
        print(f"[{nombre}] midiendo {len(imagenes)} imágenes...", flush=True)
        medido = evaluar_metodo(metodo, imagenes, config, reloj=reloj)
        metodos[nombre] = _resumen_metodo(metodo, medido, imagenes, config)
        filas += _filas_csv(nombre, medido.resultados)
        para_ejemplos[nombre] = medido.predicciones_operativas
        g = metodos[nombre]["metricas"]
        print(f"[{nombre}] TP={g['tp']} FP={g['fp']} FN={g['fn']}", flush=True)

    resumen = {
        "exp_id": exp_id,
        "conjunto": {
            "carpeta": config.conjunto_carpeta,
            "particion": config.particion,
            "imagenes": len(imagenes),
            "cajas_referencia": int(sum(len(i.cajas) for i in imagenes)),
        },
        "avisos": [AVISO_VALID] if config.particion == "valid" else [],
        "iou_min": config.iou_min,
        "dispositivo": config.dispositivo,
        "medicion": {"repeticiones": config.repeticiones, "calentamiento": config.calentamiento},
        "metodos": metodos,
        "entorno": env,
        "config_usada": config.a_diccionario(),
    }
    resumen = json.loads(json.dumps(resumen))  # solo tipos de JSON: lo devuelto = lo guardado

    carpeta.mkdir(parents=True, exist_ok=True)
    escribir_csv(carpeta / "por_imagen.csv", COLUMNAS_CSV, filas)
    _escribir_json(carpeta / "resumen.json", resumen)
    (carpeta / "tabla.md").write_text(generar_tabla_md(resumen), encoding="utf-8", newline="\n")
    for posicion, indice in enumerate(ejemplos, start=1):
        imagen = leer_imagen(imagenes[indice].ruta)
        for nombre in config.metodos:
            guardar_ejemplo(
                carpeta / "ejemplos" / nombre, posicion, imagenes[indice].nombre, imagen,
                imagenes[indice].cajas, para_ejemplos[nombre][indice],
            )  # fmt: skip

    print(f"Evaluación guardada en {mostrar(carpeta, raiz)}")
    if env.get("dirty"):
        print(
            "AVISO: el código tiene cambios sin commitear (dirty): esta ejecución no se "
            "reproduce solo con el commit."
        )
    return resumen


def crear_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.eval.localizacion",
        description="Evalúa la localización de lechugas y guarda la ejecución en runs/<exp-id>/.",
    )
    parser.add_argument(
        "--config", type=Path, default=CONFIG_POR_DEFECTO, help="YAML de la evaluación"
    )
    parser.add_argument("--exp-id", required=True, help="nombre de la carpeta runs/<exp-id>/")
    parser.add_argument(
        "--raiz", type=Path, default=RAIZ, help="raíz a la que son relativas las rutas"
    )
    return parser


def main(argv: list[str] | None = None, **inyecciones) -> int:
    """Ejecuta la evaluación. Devuelve 0 si va bien y 1 si hay un error (con mensaje saneado).

    ``inyecciones`` (``fabricas``, ``entorno``, ``reloj``) se pasan a ``ejecutar``: sirven para
    probarlo sin cargar modelos de verdad.
    """
    for flujo in (sys.stdout, sys.stderr):
        if hasattr(flujo, "reconfigure"):
            flujo.reconfigure(errors="replace")
    args = crear_parser().parse_args(argv)
    try:
        config = cargar_config_localizacion(args.config)
        ejecutar(config, args.exp_id, raiz=args.raiz, **inyecciones)
    except (ValueError, OSError, ImportError, RuntimeError, yaml.YAMLError) as error:
        print(f"ERROR: {sanear(str(error), args.raiz)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
