"""Detector genérico OWL-ViT (zero-shot): encuentra lechugas a partir de un texto.

``transformers`` y ``torch`` son opcionales: solo se importan cuando se crea el detector
(importación perezosa), así el resto del pipeline funciona sin ellos. Se instalan con
``pip install -r requirements-detector.txt``.

Es un modelo genérico, no entrenado con lechugas: puede dar falsos positivos o no
encontrar ninguna. En la v0 solo se mide, no se mejora.

Versión de transformers comprobada: 5.19.0. En ella ``OwlViTProcessor`` ya no tiene
``post_process_object_detection``; el método vigente es
``post_process_grounded_object_detection(outputs, threshold, target_sizes, text_labels)``,
que acepta ``target_sizes`` como lista de ``(alto, ancho)`` y devuelve, por imagen, un
diccionario con ``scores``, ``labels`` y ``boxes`` (tensores).
"""

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, NamedTuple

import cv2
import numpy as np

from src.pipeline.nms import nms
from src.pipeline.types import Detection

DISPOSITIVOS = ("auto", "cpu", "cuda")

_MENSAJE_INSTALAR = (
    "El detector 'owlvit' necesita la librería transformers (y PyTorch), que no está "
    "instalada. Instálala con: pip install -r requirements-detector.txt"
)


class Cargado(NamedTuple):
    """Lo que devuelve un cargador: procesador, modelo, dispositivo y contexto sin gradientes.

    ``sin_gradiente`` es una función que devuelve un gestor de contexto (``torch.no_grad``
    en el caso real) para ejecutar el modelo sin calcular gradientes.
    """

    procesador: Any
    modelo: Any
    dispositivo: str
    sin_gradiente: Callable[[], Any]


def _texto(clave: str, valor) -> str:
    if not isinstance(valor, str) or not valor.strip():
        raise ValueError(f"'owlvit.{clave}' debe ser un texto no vacío (valor: {valor!r})")
    return valor


def _fraccion(clave: str, valor) -> float:
    if isinstance(valor, bool) or not isinstance(valor, int | float) or not 0 <= valor <= 1:
        raise ValueError(f"'owlvit.{clave}' debe ser un número entre 0 y 1 (valor: {valor!r})")
    return float(valor)


@dataclass(frozen=True)
class OwlViTConfig:
    """Parámetros del detector, tal como aparecen en el bloque ``owlvit`` de configs/v0.yaml."""

    modelo: str
    consulta: str
    umbral: float
    max_detecciones: int
    nms_iou: float
    dispositivo: str

    @classmethod
    def desde_dict(cls, cfg: Mapping) -> "OwlViTConfig":
        """Lee y valida el bloque ``owlvit`` de la configuración.

        En lenguaje llano: comprueba que están todas las claves y que los valores tienen
        sentido (umbral entre 0 y 1, un máximo de al menos 1 detección...). No hay valores
        por defecto en el código: el YAML es la única fuente.

        Error: ``ValueError`` que nombra la clave que falta o que es inválida.
        """
        bloque = cfg.get("owlvit") if isinstance(cfg, Mapping) else None
        if not isinstance(bloque, Mapping):
            raise ValueError("Falta el bloque 'owlvit' en la configuración (configs/v0.yaml)")
        faltan = [clave for clave in cls.__dataclass_fields__ if clave not in bloque]
        if faltan:
            raise ValueError(f"Faltan claves en 'owlvit': {', '.join(faltan)}")

        maximo = bloque["max_detecciones"]
        if isinstance(maximo, bool) or not isinstance(maximo, int) or maximo < 1:
            raise ValueError(
                f"'owlvit.max_detecciones' debe ser un entero de al menos 1 (valor: {maximo!r})"
            )
        dispositivo = bloque["dispositivo"]
        if dispositivo not in DISPOSITIVOS:
            raise ValueError(
                f"'owlvit.dispositivo' debe ser uno de {', '.join(DISPOSITIVOS)} "
                f"(valor: {dispositivo!r})"
            )
        return cls(
            modelo=_texto("modelo", bloque["modelo"]),
            consulta=_texto("consulta", bloque["consulta"]),
            umbral=_fraccion("umbral", bloque["umbral"]),
            max_detecciones=maximo,
            nms_iou=_fraccion("nms_iou", bloque["nms_iou"]),
            dispositivo=dispositivo,
        )


def resolver_dispositivo(nombre: str, cuda_disponible: bool) -> str:
    """Decide dónde se ejecuta el modelo: ``auto`` usa la GPU si hay y, si no, la CPU.

    Pedir ``cuda`` sin GPU es un error (no se cae a CPU en silencio, para no medir otra cosa
    sin darse cuenta).
    """
    if nombre == "auto":
        return "cuda" if cuda_disponible else "cpu"
    if nombre == "cpu":
        return "cpu"
    if nombre == "cuda":
        if not cuda_disponible:
            raise ValueError(
                "Se pidió el dispositivo 'cuda' pero no hay GPU CUDA disponible; usa 'auto' o 'cpu'"
            )
        return "cuda"
    raise ValueError(f"Dispositivo '{nombre}' desconocido (válidos: {', '.join(DISPOSITIVOS)})")


def _cargar_transformers(config: OwlViTConfig) -> Cargado:
    """Carga el procesador y el modelo reales desde Hugging Face.

    Es la única función que importa ``torch`` y ``transformers``. La primera vez descarga
    el modelo (necesita conexión) y lo deja en la caché de Hugging Face.
    """
    try:
        import torch
        from transformers import OwlViTForObjectDetection, OwlViTProcessor
    except ImportError as error:
        raise ImportError(_MENSAJE_INSTALAR) from error

    dispositivo = resolver_dispositivo(config.dispositivo, torch.cuda.is_available())
    procesador = OwlViTProcessor.from_pretrained(config.modelo)
    modelo = OwlViTForObjectDetection.from_pretrained(config.modelo).to(dispositivo).eval()
    return Cargado(procesador, modelo, dispositivo, torch.no_grad)


def _a_numpy(valor) -> np.ndarray:
    """Convierte un tensor de torch (o una lista, o un array) en un array de numpy."""
    if hasattr(valor, "detach"):
        valor = valor.detach().cpu().numpy()
    return np.asarray(valor, dtype=float)


def postprocesar(
    cajas,
    puntuaciones,
    ancho: int,
    alto: int,
    umbral: float,
    nms_iou: float,
    max_detecciones: int,
) -> list[Detection]:
    """Convierte las cajas crudas del modelo en detecciones limpias.

    En lenguaje llano, en este orden: 1) se descartan las cajas con valores no finitos o
    con puntuación que no supera el umbral; 2) se recortan a los límites de la imagen
    y se redondean a píxeles enteros; 3) se descartan las degeneradas (sin ancho o sin
    alto, también las que el recorte deja sin área); 4) el NMS quita las repetidas;
    5) se ordenan de mayor a menor puntuación y 6) se conservan como mucho
    ``max_detecciones``. El tope va al final para que las repetidas no ocupen sitio.

    ``cajas`` son (x1, y1, x2, y2) en píxeles de la imagen original.
    """
    cajas = _a_numpy(cajas).reshape(-1, 4)
    puntuaciones = _a_numpy(puntuaciones).reshape(-1)

    validas = np.isfinite(cajas).all(axis=1) & np.isfinite(puntuaciones) & (puntuaciones > umbral)
    cajas, puntuaciones = cajas[validas], puntuaciones[validas]

    limites = np.array([ancho, alto, ancho, alto], dtype=float)
    cajas = np.rint(np.clip(cajas, 0, limites)).astype(int)

    con_area = (cajas[:, 2] > cajas[:, 0]) & (cajas[:, 3] > cajas[:, 1])
    cajas, puntuaciones = cajas[con_area], puntuaciones[con_area]

    conservadas = nms(cajas, puntuaciones, nms_iou)[:max_detecciones]
    return [
        Detection(box=tuple(int(v) for v in cajas[i]), score=float(puntuaciones[i]))
        for i in conservadas
    ]


class OwlViTDetector:
    """Detector de lechugas zero-shot con OWL-ViT.

    El modelo se carga UNA vez, al crear el detector, y se reutiliza en todas las imágenes.
    El tiempo de esa carga queda en ``tiempo_carga_s``, aparte del de inferencia (que mide
    el pipeline en ``timing_ms.detect``).

    ``cargador`` permite sustituir la carga real (p. ej. por un modelo falso en las pruebas);
    por defecto se cargan procesador y modelo de transformers.
    """

    def __init__(
        self, config: OwlViTConfig, cargador: Callable[[OwlViTConfig], Cargado] | None = None
    ):
        self.config = config
        cargador = cargador or _cargar_transformers
        inicio = time.perf_counter()
        cargado = cargador(config)
        self.tiempo_carga_s: float = time.perf_counter() - inicio
        self._procesador, self._modelo, self._dispositivo, self._sin_gradiente = cargado

    def detect(self, imagen: np.ndarray) -> list[Detection]:
        """Devuelve las lechugas encontradas, de mayor a menor puntuación.

        En lenguaje llano: pasa la imagen (en RGB) y el texto de la consulta al modelo; la
        librería convierte su salida en cajas en píxeles de la imagen original y aquí se
        limpian con ``postprocesar``. Si no hay nada por encima del umbral, la lista es vacía.
        """
        alto, ancho = imagen.shape[:2]
        rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
        entradas = self._procesador(
            text=[[self.config.consulta]], images=rgb, return_tensors="pt"
        ).to(self._dispositivo)
        with self._sin_gradiente():
            salidas = self._modelo(**entradas)
        resultados = self._procesador.post_process_grounded_object_detection(
            outputs=salidas, threshold=self.config.umbral, target_sizes=[(alto, ancho)]
        )
        return postprocesar(
            resultados[0]["boxes"],
            resultados[0]["scores"],
            ancho,
            alto,
            self.config.umbral,
            self.config.nms_iou,
            self.config.max_detecciones,
        )


def crear_owlvit(
    cfg: Mapping, cargador: Callable[[OwlViTConfig], Cargado] | None = None
) -> OwlViTDetector:
    """Fábrica para el registro de detectores: lee ``cfg["owlvit"]`` y crea el detector."""
    return OwlViTDetector(OwlViTConfig.desde_dict(cfg), cargador=cargador)
