"""Recortes de imagen y huellas perceptuales (dHash) para buscar duplicados.

Este módulo no sabe nada del proyecto (fuentes, etiquetas ni YAML): solo geometría de cajas,
redimensionado sin deformar y huellas de 64 bits. Lo usa ``src/data/dataset_etapa2.py``.

Las cajas de entrada van en formato COCO ``(x, y, ancho, alto)`` con (0, 0) arriba a la izquierda;
las de salida de ``ampliar_caja`` son esquinas ``(x1, y1, x2, y2)`` listas para recortar con
``imagen[y1:y2, x1:x2]``.

dHash ("difference hash"), en lenguaje llano: se pasa la imagen a gris, se reduce a 9x8 píxeles
y, en cada fila, se mira si cada píxel es más claro que el de su izquierda. Salen 8 x 8 = 64
respuestas sí/no, que se guardan como los 64 bits de un entero. Dos imágenes casi iguales (otra
compresión, otro tamaño, un poco más de brillo) dan huellas que solo difieren en unos pocos
bits; la *distancia de Hamming* cuenta cuántos bits distintos hay.
"""

import math
from collections.abc import Sequence

import cv2
import numpy as np

# numpy >= 2.0 trae una función que cuenta los bits a 1 de cada entero. Si no está (numpy 1.x),
# se cuentan con una tabla de 16 bits. Es una variable del módulo para poder probar las dos vías.
_HAY_BITWISE_COUNT = hasattr(np, "bitwise_count")
_TABLA_16_BITS = np.array([bin(i).count("1") for i in range(1 << 16)], dtype=np.uint8)

# Decimales con que se redondean las coordenadas antes de llevarlas a píxeles enteros: evita que
# el ruido de la coma flotante (0.1 * 30 = 3.0000000000000004) añada un píxel de más.
_DECIMALES = 6


# --- Geometría -----------------------------------------------------------------------------


def ampliar_caja(
    caja: Sequence[float], margen: float, ancho: int, alto: int
) -> tuple[int, int, int, int]:
    """Esquinas ``(x1, y1, x2, y2)`` de la caja ampliada con un margen y recortada a la imagen.

    En lenguaje llano: a cada lado se le añade ``margen`` veces el ancho (izquierda y derecha) o
    el alto (arriba y abajo) de la caja; con ``margen=0.10`` una caja de 20 px de ancho crece
    2 px por cada lado. Las coordenadas se redondean hacia fuera (la caja nunca se encoge) y lo
    que se sale de la imagen se corta en su borde.
    """
    x, y, w, h = (float(v) for v in caja)
    x1 = math.floor(round(x - margen * w, _DECIMALES))
    y1 = math.floor(round(y - margen * h, _DECIMALES))
    x2 = math.ceil(round(x + w + margen * w, _DECIMALES))
    y2 = math.ceil(round(y + h + margen * h, _DECIMALES))
    return max(0, x1), max(0, y1), min(int(ancho), x2), min(int(alto), y2)


def motivo_caja_rara(
    caja: Sequence[float], ancho: int, alto: int, min_lado_px: float
) -> str | None:
    """Por qué una caja no sirve para recortar, o ``None`` si sirve.

    Motivos: ``degenerada`` (ancho o alto cero o negativo), ``fuera de la imagen`` (no toca la
    imagen en ningún punto) y ``pequeña`` (su lado menor no llega a ``min_lado_px``: ampliarla
    a 512 px no aportaría detalle). Una caja que solo se sale en parte no es rara: se recorta al
    borde (ver ``se_sale_en_parte``).
    """
    x, y, w, h = (float(v) for v in caja)
    if w <= 0 or h <= 0:
        return "degenerada"
    if x >= ancho or y >= alto or x + w <= 0 or y + h <= 0:
        return "fuera de la imagen"
    if min(w, h) < min_lado_px:
        return "pequeña"
    return None


def se_sale_en_parte(caja: Sequence[float], ancho: int, alto: int) -> bool:
    """True si la caja pasa del borde de la imagen por algún lado (justo en el borde no cuenta)."""
    x, y, w, h = (float(v) for v in caja)
    return x < 0 or y < 0 or x + w > ancho or y + h > alto


def redimensionar_lado_mayor(imagen: np.ndarray, lado: int) -> np.ndarray:
    """Copia de la imagen con su lado mayor de ``lado`` píxeles, sin deformarla.

    En lenguaje llano: los dos lados se multiplican por el mismo factor (``lado`` / lado mayor) y
    se redondean a píxeles. Para reducir se usa ``INTER_AREA`` (promedia los píxeles y no deja
    dientes de sierra); para ampliar, ``INTER_CUBIC`` (más suave que la lineal).
    """
    alto, ancho = imagen.shape[:2]
    escala = lado / max(alto, ancho)
    nuevo = (max(1, round(ancho * escala)), max(1, round(alto * escala)))
    if nuevo == (ancho, alto):
        return imagen.copy()
    interpolacion = cv2.INTER_AREA if escala < 1 else cv2.INTER_CUBIC
    return cv2.resize(imagen, nuevo, interpolation=interpolacion)


# --- Huellas -------------------------------------------------------------------------------


def dhash(imagen: np.ndarray) -> int:
    """Huella dHash de 64 bits de una imagen (BGR o gris), como entero de 0 a 2**64 - 1.

    En lenguaje llano: gris, reducción a 9 columnas x 8 filas y, por cada fila, 8 comparaciones
    "¿es este píxel más claro que el de su izquierda?". Los 64 resultados se leen de arriba abajo
    y de izquierda a derecha y forman los bits del número, del más significativo al menos.
    Una imagen de un solo color da 0.
    """
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY) if imagen.ndim == 3 else imagen
    pequena = cv2.resize(gris, (9, 8), interpolation=cv2.INTER_AREA).astype(np.int16)
    bits = (pequena[:, 1:] > pequena[:, :-1]).flatten()
    valor = 0
    for bit in bits:
        valor = (valor << 1) | int(bit)
    return valor


def huellas(imagen: np.ndarray, giros_y_espejos: bool) -> tuple[int, ...]:
    """La huella directa de la imagen y, si se pide, las de sus giros y espejos.

    En lenguaje llano: dHash cambia por completo si la foto se gira o se voltea, así que una copia
    girada no se reconocería. Con ``giros_y_espejos`` se calculan las 8 huellas de la imagen
    girada 0, 90, 180 y 270 grados, cada una con y sin espejo (el grupo diédrico de un cuadrado).
    La primera huella es siempre la de la imagen tal cual.
    """
    if not giros_y_espejos:
        return (dhash(imagen),)
    variantes = []
    for giros in range(4):
        girada = np.rot90(imagen, giros)
        for vista in (girada, np.fliplr(girada)):
            variantes.append(dhash(np.ascontiguousarray(vista)))
    return tuple(variantes)


def contar_bits(valores: np.ndarray, *, con_tabla: bool = False) -> np.ndarray:
    """Número de bits a 1 de cada entero de un array ``uint64`` (resultado ``uint8``).

    En lenguaje llano: con numpy >= 2.0 se usa ``np.bitwise_count``; con ``con_tabla`` (o si esa
    función no existe) cada número se parte en cuatro trozos de 16 bits y se suma lo que dice una
    tabla precalculada para cada trozo.
    """
    if not con_tabla and _HAY_BITWISE_COUNT:
        return np.bitwise_count(valores)
    total = np.zeros(valores.shape, dtype=np.uint8)
    for desplazamiento in (0, 16, 32, 48):
        trozo = (valores >> np.uint64(desplazamiento)) & np.uint64(0xFFFF)
        total += _TABLA_16_BITS[trozo.astype(np.intp)]
    return total


def pares_cercanos(
    huellas_por_imagen: Sequence[Sequence[int]], umbral: int, bloque: int = 256
) -> list[tuple[int, int, int]]:
    """Pares ``(i, j, distancia)``, con ``i < j``, a distancia de Hamming <= ``umbral``.

    En lenguaje llano: ``huellas_por_imagen[i]`` son las huellas de la imagen ``i`` (la directa
    primero y, si se calcularon, las de sus giros y espejos; todas con el mismo número). La
    distancia entre dos imágenes es la menor entre la huella directa de una y cualquier huella de
    la otra, mirando en los dos sentidos: así una copia girada queda a distancia 0.

    Se compara todo con todo, pero por bloques de ``bloque`` filas y con operaciones de numpy
    (XOR y contar bits), para no ocupar n x n en memoria. Con unas 20 000 imágenes basta.
    """
    n = len(huellas_por_imagen)
    if n < 2:
        return []
    todas = np.array([list(h) for h in huellas_por_imagen], dtype=np.uint64)  # (n, variantes)
    directas = todas[:, 0]
    encontrados: dict[tuple[int, int], int] = {}
    for inicio in range(0, n, bloque):
        filas = directas[inicio : inicio + bloque, None]  # (b, 1)
        minimo = np.full((filas.shape[0], n), 64, dtype=np.uint8)
        for variante in range(todas.shape[1]):
            distancia = contar_bits(np.bitwise_xor(filas, todas[None, :, variante]))
            np.minimum(minimo, distancia, out=minimo)
        for fila, j in zip(*np.nonzero(minimo <= umbral), strict=True):
            i = inicio + int(fila)
            j = int(j)
            if i == j:
                continue
            clave = (min(i, j), max(i, j))
            valor = int(minimo[fila, j])
            encontrados[clave] = min(encontrados.get(clave, valor), valor)
    return sorted((i, j, d) for (i, j), d in encontrados.items())


def agrupar(n: int, pares: Sequence[tuple[int, int, int]]) -> list[list[int]]:
    """Grupos de duplicados: las cadenas de pares unidas (A~B y B~C forman un grupo A, B, C).

    En lenguaje llano: es un "union-find". Cada imagen empieza sola; por cada par se juntan los
    grupos de sus dos imágenes. Al final se devuelven solo los grupos con dos o más imágenes, cada
    uno ordenado y la lista ordenada por su primer elemento.
    """
    padre = list(range(n))

    def raiz(i: int) -> int:
        while padre[i] != i:
            padre[i] = padre[padre[i]]  # acorta el camino para la próxima vez
            i = padre[i]
        return i

    for i, j, _ in pares:
        a, b = raiz(i), raiz(j)
        if a != b:
            padre[max(a, b)] = min(a, b)
    grupos: dict[int, list[int]] = {}
    for i in range(n):
        grupos.setdefault(raiz(i), []).append(i)
    return sorted((sorted(g) for g in grupos.values() if len(g) > 1), key=lambda g: g[0])
