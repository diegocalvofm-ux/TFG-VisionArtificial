"""Métricas de la localización: precisión, recall, F1, IoU medio, AP (pycocotools) y latencias."""

import contextlib
import io
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from src.utils.profiling import mediana


def prf(tp: int, fp: int, fn: int) -> dict[str, float | None]:
    """Precisión, recall y F1 a partir de verdaderos positivos, falsos positivos y falsos negativos.

    En lenguaje llano: la precisión dice qué fracción de lo que el método marcó era una lechuga
    (TP / (TP + FP)); el recall, qué fracción de las lechugas reales encontró (TP / (TP + FN)); y
    el F1 resume las dos (2·TP / (2·TP + FP + FN)). Si el cociente tiene denominador 0 (por
    ejemplo la precisión de un método que no marcó nada) el valor es ``None``, no 0: no está
    definido.
    """

    def cociente(numerador: int, denominador: int) -> float | None:
        return numerador / denominador if denominador > 0 else None

    return {
        "precision": cociente(tp, tp + fp),
        "recall": cociente(tp, tp + fn),
        "f1": cociente(2 * tp, 2 * tp + fp + fn),
    }


@dataclass(frozen=True)
class ResultadoImagen:
    """Lo medido en una imagen con un método: conteos, IoU de los pares y latencia."""

    nombre: str
    n_gt: int
    n_pred: int  # predicciones que superan el umbral operativo
    tp: int
    fp: int
    fn: int
    ious: list[float]  # IoU de cada par emparejado
    latencia_ms: float

    @property
    def sin_deteccion(self) -> bool:
        """``True`` si el método no dejó ninguna predicción en la imagen."""
        return self.n_pred == 0

    @property
    def iou_medio(self) -> float | None:
        return float(np.mean(self.ious)) if self.ious else None

    @property
    def precision(self) -> float | None:
        return prf(self.tp, self.fp, self.fn)["precision"]

    @property
    def recall(self) -> float | None:
        return prf(self.tp, self.fp, self.fn)["recall"]

    @property
    def f1(self) -> float | None:
        return prf(self.tp, self.fp, self.fn)["f1"]


def agregar(resultados: Sequence[ResultadoImagen]) -> dict:
    """Métricas sobre todo el conjunto: suma los conteos de todas las imágenes.

    En lenguaje llano: no se promedian las métricas de cada imagen; se suman los aciertos y los
    errores de todas y se calcula la precisión, el recall y el F1 una vez (así una imagen con
    muchas lechugas pesa más). El IoU medio es la media sobre TODOS los pares emparejados.

    Error: ``ValueError`` si no hay ninguna imagen.
    """
    if not resultados:
        raise ValueError("No hay resultados que agregar: la lista está vacía")
    tp = sum(r.tp for r in resultados)
    fp = sum(r.fp for r in resultados)
    fn = sum(r.fn for r in resultados)
    ious = [valor for r in resultados for valor in r.ious]
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        **prf(tp, fp, fn),
        "iou_medio": float(np.mean(ious)) if ious else None,
        "imagenes": len(resultados),
        "imagenes_sin_deteccion": sum(r.sin_deteccion for r in resultados),
    }


def ap_coco(
    cajas_ref: Sequence[np.ndarray],
    predicciones: Sequence[tuple[np.ndarray, np.ndarray]],
    tamanos: Sequence[tuple[int, int]],
) -> tuple[float | None, float | None]:
    """AP50 y AP50:95 de una clase con pycocotools (``COCOeval`` de cajas).

    En lenguaje llano: el AP (precisión media) resume en un número cuánto acierta un método con
    TODAS sus predicciones ordenadas por confianza, sin elegir un umbral. AP50 exige un IoU de
    0,5 para dar un acierto y AP50:95 promedia los umbrales de 0,50 a 0,95 en pasos de 0,05.
    Se usa la implementación de referencia de COCO para que la cifra sea comparable.

    ``cajas_ref[i]`` es una matriz (N, 4) con las cajas de referencia de la imagen ``i``;
    ``predicciones[i]`` es un par ``(cajas (M, 4), confianzas (M,))`` y ``tamanos[i]`` es
    ``(ancho, alto)``. Todas las cajas son ``(x1, y1, x2, y2)``. Sin ninguna predicción en todo el
    conjunto el AP es 0,0; sin ninguna caja de referencia no está definido y devuelve
    ``(None, None)``.

    Error: ``ImportError`` si pycocotools no está instalado.
    """
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as error:
        raise ImportError(
            "El cálculo del AP necesita pycocotools. Instálalo con: pip install pycocotools "
            "(está fijado en requirements-train.txt)"
        ) from error

    imagenes, referencias, resultados = [], [], []
    for i, (cajas, (ancho, alto)) in enumerate(zip(cajas_ref, tamanos, strict=True), start=1):
        imagenes.append({"id": i, "width": int(ancho), "height": int(alto)})
        for x1, y1, x2, y2 in np.asarray(cajas, dtype=float).reshape(-1, 4):
            referencias.append(
                {
                    "id": len(referencias) + 1, "image_id": i, "category_id": 1, "iscrowd": 0,
                    "bbox": [x1, y1, x2 - x1, y2 - y1], "area": (x2 - x1) * (y2 - y1),
                }
            )  # fmt: skip
    if not referencias:
        return None, None
    for i, (cajas, confianzas) in enumerate(predicciones, start=1):
        for (x1, y1, x2, y2), confianza in zip(
            np.asarray(cajas, dtype=float).reshape(-1, 4), np.asarray(confianzas, dtype=float)
        ):
            resultados.append(
                {
                    "image_id": i, "category_id": 1, "score": float(confianza),
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                }
            )  # fmt: skip
    if not resultados:
        return 0.0, 0.0

    with contextlib.redirect_stdout(io.StringIO()):  # pycocotools imprime mucho
        referencia = COCO()
        referencia.dataset = {
            "images": imagenes,
            "annotations": referencias,
            "categories": [{"id": 1, "name": "lechuga"}],
        }
        referencia.createIndex()
        evaluador = COCOeval(referencia, referencia.loadRes(resultados), iouType="bbox")
        evaluador.evaluate()
        evaluador.accumulate()
        evaluador.summarize()
    ap50_95, ap50 = (float(evaluador.stats[0]), float(evaluador.stats[1]))
    return tuple(None if v < 0 else v for v in (ap50, ap50_95))


def percentil(valores: Sequence[float], q: float) -> float:
    """Percentil ``q`` (0-100) de una lista, interpolando entre valores vecinos.

    En lenguaje llano: el percentil 90 es el valor por debajo del cual queda el 90 % de los
    datos; sirve para ver el tiempo "malo pero no extremo" que ignora la mediana.

    Error: ``ValueError`` si la lista está vacía.
    """
    valores = list(valores)
    if not valores:
        raise ValueError("No se puede calcular un percentil de una lista vacía")
    return float(np.percentile(valores, q))


def resumen_latencia(ms_por_imagen: Sequence[float]) -> dict[str, float]:
    """Mediana y percentil 90 de las latencias por imagen (cada una, mediana de varias pasadas)."""
    return {"mediana_ms": mediana(ms_por_imagen), "p90_ms": percentil(ms_por_imagen, 90)}
