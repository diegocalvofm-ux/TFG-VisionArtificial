"""Salidas de la evaluación: CSV por imagen, tabla de Markdown e imágenes de ejemplo.

Ninguna salida lleva rutas absolutas: el CSV solo tiene nombres de archivo y el resto recibe rutas
relativas ya saneadas.
"""

import csv
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from src.pipeline.draw import texto_ascii
from src.pipeline.imagenes import guardar_imagen
from src.pipeline.types import Detection

# Colores BGR: la referencia en verde y lo predicho en rojo, para distinguirlos de un vistazo.
COLOR_REFERENCIA = (0, 200, 0)
COLOR_PREDICHA = (0, 0, 255)

_FUENTE = cv2.FONT_HERSHEY_SIMPLEX

# Aviso fijo sobre la partición ``valid``: el checkpoint de RF-DETR se eligió con ella.
AVISO_VALID = (
    "La partición `valid` se usó para elegir el checkpoint de RF-DETR (`checkpoint_best_total`), "
    "así que sus métricas son optimistas; no hay una partición de test independiente."
)


def formato(valor: float | None, decimales: int = 3, ausente: str = "—") -> str:
    """Número con coma decimal (para la memoria); ``ausente`` si no está definido (``None``)."""
    if valor is None:
        return ausente
    return f"{valor:.{decimales}f}".replace(".", ",")


def escribir_csv(ruta: Path, columnas: Sequence[str], filas: Sequence[dict]) -> None:
    """Escribe un CSV en UTF-8; lo que vale ``None`` (no definido) se deja vacío."""
    with Path(ruta).open("w", encoding="utf-8", newline="") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=list(columnas), lineterminator="\n")
        escritor.writeheader()
        for fila in filas:
            escritor.writerow({c: "" if fila.get(c) is None else fila[c] for c in columnas})


def dibujar_cajas(
    imagen: np.ndarray, cajas_ref: Sequence[Sequence[float]], predicciones: Sequence[Detection]
) -> np.ndarray:
    """Copia de la imagen con las cajas de referencia en verde y las predichas en rojo.

    En lenguaje llano: la original no se modifica. Cada predicción lleva su puntuación escrita
    encima de la caja (la línea base, que no tiene confianza, no la muestra). Las cajas que se
    salen de la imagen se recortan a su borde; el grosor crece con el tamaño de la imagen.
    """
    salida = imagen.copy()
    alto, ancho = salida.shape[:2]
    escala = max(0.5, max(alto, ancho) / 800)
    grosor = max(1, round(escala * 2))

    def recortar(x1, y1, x2, y2):
        return (
            (min(max(int(x1), 0), ancho - 1), min(max(int(y1), 0), alto - 1)),
            (min(max(int(x2), 0), ancho - 1), min(max(int(y2), 0), alto - 1)),
        )

    for caja in cajas_ref:
        esquina_1, esquina_2 = recortar(*caja)
        cv2.rectangle(salida, esquina_1, esquina_2, COLOR_REFERENCIA, grosor)
    for deteccion in predicciones:
        esquina_1, esquina_2 = recortar(*deteccion.box)
        cv2.rectangle(salida, esquina_1, esquina_2, COLOR_PREDICHA, grosor)
        if deteccion.score < 1.0:  # 1,0 es el relleno de la línea base: sin confianza real
            texto = texto_ascii(f"{deteccion.score:.2f}")
            origen = (esquina_1[0], max(esquina_1[1] - 4, 12))
            cv2.putText(salida, texto, origen, _FUENTE, escala * 0.6, COLOR_PREDICHA, 1)
    return salida


def guardar_ejemplo(
    carpeta: Path,
    posicion: int,
    nombre_imagen: str,
    imagen: np.ndarray,
    cajas_ref: Sequence[Sequence[float]],
    predicciones: Sequence[Detection],
) -> None:
    """Guarda ``<carpeta>/<NN>_<nombre>.jpg`` con las cajas dibujadas (crea la carpeta)."""
    carpeta.mkdir(parents=True, exist_ok=True)
    nombre = f"{posicion:02d}_{Path(nombre_imagen).stem}.jpg"
    guardar_imagen(carpeta / nombre, dibujar_cajas(imagen, cajas_ref, predicciones))


def _celdas(nombre: str, m: dict) -> list[str]:
    g = m["metricas"]
    lat = m["latencia"]
    umbral = m["umbral"]
    return [
        nombre,
        formato(umbral, 2, "—"),
        formato(g["precision"]),
        formato(g["recall"]),
        formato(g["f1"]),
        formato(g["iou_medio"]),
        formato(m["ap50"], 3, "n/a"),
        formato(m["ap50_95"], 3, "n/a"),
        f"{m['imagenes_sin_deteccion']}/{g['imagenes']}",
        formato(lat["mediana_ms"], 1),
        formato(lat["p90_ms"], 1),
        formato(m["carga_modelo_s"], 2),
    ]


def _parametro(valor) -> str:
    if isinstance(valor, float):
        return formato(valor, 4).rstrip("0").rstrip(",") or "0"
    if isinstance(valor, list):
        return "[" + ", ".join(_parametro(v) for v in valor) + "]"
    return str(valor)


def generar_tabla_md(resumen: dict) -> str:
    """Tabla de Markdown lista para la memoria, con los valores usados y las notas de lectura.

    En lenguaje llano: una fila por método con las métricas globales y la latencia, y debajo la
    configuración de cada método y los avisos que hay que tener en cuenta al leer los números
    (partición usada, parámetros no ajustados, AP que no aplica, cómo se calcula el p90...).
    """
    metodos = resumen["metodos"]
    conjunto = resumen["conjunto"]
    medicion = resumen["medicion"]
    cabecera = [
        "Método", "Umbral", "Precisión", "Recall", "F1", "IoU medio", "AP50", "AP50:95",
        "Sin detección", "Mediana (ms)", "P90 (ms)", "Carga (s)",
    ]  # fmt: skip
    lineas = [
        f"Localización de lechugas en la partición `{conjunto['particion']}` de "
        f"`{conjunto['carpeta']}` ({conjunto['imagenes']} imágenes, "
        f"{conjunto['cajas_referencia']} cajas de referencia).",
        "",
        "| " + " | ".join(cabecera) + " |",
        "|" + "|".join(["---"] + ["---:"] * (len(cabecera) - 1)) + "|",
    ]
    lineas += ["| " + " | ".join(_celdas(n, m)) + " |" for n, m in metodos.items()]

    lineas += ["", "Valores usados:", ""]
    for nombre, m in metodos.items():
        pares = {"umbral": m["umbral"], "umbral_ap": m["umbral_ap"], **m["parametros"]}
        pares = {k: v for k, v in pares.items() if v is not None}
        detalle = "; ".join(f"{k} = {_parametro(v)}" for k, v in pares.items())
        lineas.append(f"- **{nombre}**: {detalle}")

    iou_min = _parametro(resumen["iou_min"])
    lineas += [
        "",
        "Notas:",
        "",
        f"- Un acierto exige IoU >= {iou_min}. El emparejamiento es voraz: de mayor a menor "
        "confianza, cada predicción toma la caja de referencia libre con más IoU. Precisión, "
        "recall, F1, IoU medio y «sin detección» (imágenes sin ninguna predicción) se calculan "
        "con las predicciones que superan el umbral operativo.",
        "- AP50 y AP50:95 (pycocotools) usan todas las predicciones por encima del umbral bajo "
        "`umbral_ap`, sin elegir umbral operativo.",
        f"- Latencia en {resumen['dispositivo'].upper()} por imagen (de la imagen en memoria a las "
        f"cajas, incluido el pre y el postproceso del método): {medicion['repeticiones']} "
        f"repeticiones tras {medicion['calentamiento']} de calentamiento. La mediana y el p90 son "
        "estadísticos de las medianas por imagen. La carga del modelo se mide aparte.",
    ]
    if "owlvit" in metodos:
        lineas.append(
            "- OWL-ViT no está entrenado con lechugas. Se anota el tope de cajas por imagen usado "
            f"({metodos['owlvit']['parametros'].get('max_detecciones')}): el de la v0 (10) "
            "limitaría el recall."
        )
    if "rfdetr" in metodos:
        lineas.append(
            f"- El umbral operativo de RF-DETR ({formato(metodos['rfdetr']['umbral'], 2)}) está "
            "fijado de antemano, no ajustado sobre esta partición."
        )
    if "linea_base_hsv" in metodos:
        lineas.append(
            "- Los parámetros de la línea base HSV no están ajustados para este conjunto (son "
            "valores por defecto fijados de antemano). No tiene puntuación de confianza: el AP "
            "no aplica (n/a)."
        )
    if conjunto["particion"] == "valid":
        lineas.append(f"- {AVISO_VALID}")
    return "\n".join(lineas) + "\n"
