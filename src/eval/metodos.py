"""Adaptadores de los tres métodos de localización a una interfaz común.

Cada método se convierte en un ``Metodo``: una función ``predecir(imagen_BGR)`` que devuelve
``Detection`` (caja y puntuación) más lo que la evaluación necesita saber de él (si tiene
confianza, sus umbrales, el tiempo de carga y los parámetros usados, que se anotan en la tabla).

OWL-ViT reutiliza el detector de la v0 (``src.pipeline.detectors.owlvit``) sin tocarlo: misma
consulta y misma carga. RF-DETR se carga desde un checkpoint con la API de rfdetr 1.11.2.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np

from src.config import load_config
from src.eval.config import ConfigLocalizacion
from src.eval.linea_base import LineaBaseHSV
from src.pipeline.detectors.owlvit import Cargado, OwlViTConfig, OwlViTDetector
from src.pipeline.types import Detection


@dataclass(frozen=True)
class Metodo:
    """Un método de localización listo para evaluar.

    ``umbral`` es el umbral operativo (lo que se cuenta para precisión, recall y F1: solo valen
    las predicciones con puntuación MAYOR) y ``umbral_ap`` el umbral bajo con el que se ejecuta el
    modelo para calcular el AP. Ambos son ``None`` si el método no tiene confianza.
    ``tiempo_carga_s`` es ``None`` si no hay modelo que cargar.
    """

    tiene_confianza: bool
    predecir: Callable[[np.ndarray], list[Detection]]
    tiempo_carga_s: float | None
    umbral: float | None
    umbral_ap: float | None
    parametros: dict


def crear_owlvit(
    cfg: ConfigLocalizacion, raiz: Path, cargador: Callable[[OwlViTConfig], Cargado] | None = None
) -> Metodo:
    """OWL-ViT sin entrenar, con los valores de ``configs/v0.yaml`` salvo lo que diga el YAML.

    En lenguaje llano: lee el bloque ``owlvit`` de la v0 (modelo, consulta, umbral, tope de cajas y
    NMS) y lo adapta: la consulta, el umbral operativo y el tope de cajas del YAML de evaluación
    sustituyen a los de la v0 cuando no son ``null``, y el dispositivo es el de la evaluación (CPU).
    El modelo se ejecuta con el umbral bajo ``umbral_ap``, de modo que de una sola pasada salen las
    predicciones para el AP y, filtrando por ``umbral``, las del punto de operación (da lo mismo
    que ejecutarlo con ``umbral``: el NMS y el tope solo dependen de las cajas mejor puntuadas).

    Errores: ``ValueError`` si el umbral operativo es menor que ``umbral_ap``; los de la lectura de
    ``configs/v0.yaml``. ``cargador`` permite sustituir la carga real al probar.
    """
    base = OwlViTConfig.desde_dict(load_config(raiz / cfg.owlvit_config_base))
    consulta = cfg.owlvit_consulta or base.consulta
    umbral = base.umbral if cfg.owlvit_umbral is None else cfg.owlvit_umbral
    maximo = cfg.owlvit_max_detecciones or base.max_detecciones
    if umbral < cfg.owlvit_umbral_ap:
        raise ValueError(
            f"owlvit.umbral ({umbral}) no puede ser menor que owlvit.umbral_ap "
            f"({cfg.owlvit_umbral_ap})"
        )
    config = replace(
        base,
        consulta=consulta,
        umbral=cfg.owlvit_umbral_ap,
        max_detecciones=maximo,
        dispositivo=cfg.dispositivo,
    )
    detector = OwlViTDetector(config, cargador=cargador)
    return Metodo(
        tiene_confianza=True,
        predecir=detector.detect,
        tiempo_carga_s=detector.tiempo_carga_s,
        umbral=umbral,
        umbral_ap=cfg.owlvit_umbral_ap,
        parametros={
            "modelo": base.modelo,
            "consulta": consulta,
            "max_detecciones": maximo,
            "nms_iou": base.nms_iou,
            "dispositivo": detector.dispositivo,
        },
    )


def _cargar_rfdetr(ruta: Path, dispositivo: str):
    """Carga RF-DETR desde un checkpoint de entrenamiento (única función que importa rfdetr).

    ``RFDETR.from_checkpoint`` (rfdetr 1.11.2) deduce la clase (Nano, Small...) del propio
    checkpoint. El dispositivo guardado en él es el del entrenamiento, así que se pasa el de la
    evaluación.
    """
    try:
        from rfdetr import RFDETR
    except ImportError as error:
        raise ImportError(
            "No está instalado rfdetr. Instálalo con `pip install -c <constraints> -r "
            "requirements-train.txt`"
        ) from error
    return RFDETR.from_checkpoint(str(ruta), device=dispositivo)


def crear_rfdetr(
    cfg: ConfigLocalizacion, raiz: Path, cargador: Callable[[Path, str], object] | None = None
) -> Metodo:
    """RF-DETR con los pesos de ``rfdetr.pesos``.

    En lenguaje llano: carga el checkpoint una vez (y anota cuánto tarda) y, por cada imagen, la
    pasa a RGB (rfdetr la espera así; el pipeline usa BGR), llama a ``predict`` con el umbral bajo
    ``umbral_ap`` y devuelve las cajas redondeadas a píxeles, de mayor a menor puntuación.

    Errores: ``FileNotFoundError`` (con la ruta relativa) si no existen los pesos;
    ``ImportError`` si falta rfdetr. ``cargador(ruta, dispositivo)`` se puede sustituir al probar.
    """
    ruta = raiz / cfg.rfdetr_pesos
    if cargador is None:
        if not ruta.is_file():
            raise FileNotFoundError(f"No existen los pesos de RF-DETR: {cfg.rfdetr_pesos}")
        cargador = _cargar_rfdetr
    inicio = time.perf_counter()
    modelo = cargador(ruta, cfg.dispositivo)
    carga = time.perf_counter() - inicio

    def predecir(imagen: np.ndarray) -> list[Detection]:
        rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
        resultado = modelo.predict(rgb, threshold=cfg.rfdetr_umbral_ap, include_source_image=False)
        cajas = np.rint(np.asarray(resultado.xyxy, dtype=float).reshape(-1, 4)).astype(int)
        puntuaciones = np.asarray(resultado.confidence, dtype=float).reshape(-1)
        orden = np.argsort(-puntuaciones, kind="stable")
        return [
            Detection(box=tuple(int(v) for v in cajas[i]), score=float(puntuaciones[i]))
            for i in orden
        ]

    return Metodo(
        tiene_confianza=True,
        predecir=predecir,
        tiempo_carga_s=carga,
        umbral=cfg.rfdetr_umbral,
        umbral_ap=cfg.rfdetr_umbral_ap,
        parametros={"pesos": cfg.rfdetr_pesos, "dispositivo": cfg.dispositivo},
    )


def crear_linea_base(cfg: ConfigLocalizacion) -> Metodo:
    """La línea base HSV: sin modelo que cargar y sin confianza (AP no aplica)."""
    p = cfg.hsv
    return Metodo(
        tiene_confianza=False,
        predecir=LineaBaseHSV(p).detect,
        tiempo_carga_s=None,
        umbral=None,
        umbral_ap=None,
        parametros={
            "h": list(p.h), "s": list(p.s), "v": list(p.v),
            "apertura_px": p.apertura_px, "cierre_px": p.cierre_px,
            "area_min_frac": p.area_min_frac,
        },
    )  # fmt: skip


def crear_fabricas(cfg: ConfigLocalizacion, raiz: Path) -> dict[str, Callable[[], Metodo]]:
    """Una fábrica por método: crearlos de uno en uno evita cargar un modelo hasta que toca."""
    return {
        "owlvit": lambda: crear_owlvit(cfg, raiz),
        "rfdetr": lambda: crear_rfdetr(cfg, raiz),
        "linea_base_hsv": lambda: crear_linea_base(cfg),
    }
