"""Dibujo de la imagen anotada: la caja de cada lechuga y su nivel global."""

from collections.abc import Sequence

import cv2
import numpy as np

from src.data.schema import GLOBAL_LEVELS

# Color BGR de cada nivel, en el mismo orden que GLOBAL_LEVELS (ACCEPT, REVIEW, NON-CONFORMING):
# verde, naranja y rojo.
COLORES: tuple[tuple[int, int, int], ...] = ((0, 200, 0), (0, 165, 255), (0, 0, 255))

_FUENTE = cv2.FONT_HERSHEY_SIMPLEX


def texto_ascii(texto: str) -> str:
    """Cambia por ``?`` lo que no sea ASCII: las fuentes de OpenCV no pintan tildes ni eñes."""
    return texto.encode("ascii", "replace").decode("ascii")


def dibujar(imagen: np.ndarray, lettuces: Sequence[dict]) -> np.ndarray:
    """Devuelve una copia de la imagen con la caja y el nivel de cada lechuga dibujados.

    En lenguaje llano: por cada lechuga se dibuja un rectángulo del color de su nivel
    (verde, naranja o rojo) y, sobre la esquina superior izquierda, un rótulo con el nivel
    y la puntuación. La imagen original no se modifica. El tamaño de letra y de línea
    crece con la imagen para que se vea igual en una foto pequeña que en una grande.

    ``lettuces`` es la lista del resultado del pipeline (cada una con ``box``, ``score`` y
    ``global``). Las cajas que se salen de la imagen se recortan a su borde.
    """
    salida = imagen.copy()
    alto, ancho = salida.shape[:2]
    escala = max(0.5, max(alto, ancho) / 800)
    grosor_caja = max(1, round(escala * 2))
    grosor_texto = max(1, round(escala))

    for lechuga in lettuces:
        x1, y1, x2, y2 = lechuga["box"]
        x1, x2 = (min(max(x, 0), ancho - 1) for x in (x1, x2))
        y1, y2 = (min(max(y, 0), alto - 1) for y in (y1, y2))
        nivel = lechuga["global"]
        color = COLORES[GLOBAL_LEVELS.index(nivel)]
        cv2.rectangle(salida, (x1, y1), (x2, y2), color, grosor_caja)

        texto = texto_ascii(f"{nivel} {lechuga['score']:.2f}")
        (ancho_texto, alto_texto), base = cv2.getTextSize(texto, _FUENTE, escala, grosor_texto)
        if y1 - alto_texto - base - 2 >= 0:  # cabe encima de la caja
            y_texto = y1 - base - 2
        else:  # si no, dentro, bajo el borde superior
            y_texto = y1 + alto_texto + grosor_caja + 2
        x_texto = min(x1, max(ancho - ancho_texto - 1, 0))
        # Fondo del color del nivel y letra negra, para que se lea sobre cualquier foto.
        cv2.rectangle(
            salida,
            (x_texto, y_texto - alto_texto - 2),
            (x_texto + ancho_texto, y_texto + base),
            color,
            cv2.FILLED,
        )
        cv2.putText(salida, texto, (x_texto, y_texto), _FUENTE, escala, (0, 0, 0), grosor_texto)
    return salida
