"""Transformaciones de las imágenes de la etapa 2 y el ``Dataset`` de PyTorch.

En evaluación (valid, test y ``predecir``) cada recorte solo se encaja en un cuadrado y se
normaliza. En train, además, se aplican aumentos aleatorios, que son **solo geométricos**:
recorte aleatorio (sin deformar, con la configuración acordada), volteos, giros de 90 grados y una
rotación pequeña. Como mucho se añaden brillo y contraste suaves, que cambian los tres canales por
igual. **Nunca** se cambia el tono ni la saturación: el grupo ``color`` (amarilleo, pardeamiento)
se juzga precisamente por el color, y un aumento así enseñaría al modelo que un recorte amarillento
puede ser ``COLOR_FRESH_OK``.

Se usa ``torchvision.transforms.v2`` sobre tensores ``uint8`` en formato (canales, alto, ancho).
"""

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F

from src.pipeline.imagenes import leer_imagen
from src.train.etapa2.config import Aumentos

# Normalización de ImageNet: la de los pesos preentrenados (modelo.py comprueba que timm usa esta).
IMAGENET_MEDIA = (0.485, 0.456, 0.406)
IMAGENET_DESV = (0.229, 0.224, 0.225)
# Color con que se rellena alrededor del recorte: la media de ImageNet, que tras normalizar es 0.
COLOR_RELLENO = tuple(round(m * 255) for m in IMAGENET_MEDIA)


def a_tensor_rgb(imagen_bgr: np.ndarray) -> torch.Tensor:
    """De la imagen BGR de OpenCV (alto, ancho, 3) a un tensor RGB ``uint8`` (3, alto, ancho)."""
    rgb = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).contiguous()


def cuadrar(imagen: torch.Tensor, lado: int) -> torch.Tensor:
    """Encaja la imagen en un cuadrado de ``lado`` píxeles sin deformarla ni cortarla.

    En lenguaje llano: el lado mayor pasa a medir ``lado`` (el otro, en proporción) y lo que falta
    hasta el cuadrado se rellena por igual a los dos lados con el color medio de ImageNet. Así la
    lechuga conserva su forma y no se pierde ningún borde, donde puede estar un defecto (el tallo,
    la base del corte).
    """
    alto, ancho = imagen.shape[-2:]
    if (alto, ancho) != (lado, lado):
        escala = lado / max(alto, ancho)
        nuevo = (max(1, round(alto * escala)), max(1, round(ancho * escala)))
        imagen = F.resize(imagen, list(nuevo), antialias=True)
    alto, ancho = imagen.shape[-2:]
    if (alto, ancho) == (lado, lado):
        return imagen
    arriba, izquierda = (lado - alto) // 2, (lado - ancho) // 2
    salida = torch.empty((imagen.shape[0], lado, lado), dtype=imagen.dtype)
    salida[:] = torch.tensor(COLOR_RELLENO, dtype=imagen.dtype).view(-1, 1, 1)
    salida[:, arriba : arriba + alto, izquierda : izquierda + ancho] = imagen
    return salida


class Cuadrar:
    """``cuadrar`` como paso de una ``v2.Compose`` (una clase y no una lambda: así se puede
    enviar a los procesos del ``DataLoader`` también en Windows)."""

    def __init__(self, lado: int):
        self.lado = lado

    def __call__(self, imagen: torch.Tensor) -> torch.Tensor:
        return cuadrar(imagen, self.lado)

    def __repr__(self) -> str:
        return f"Cuadrar(lado={self.lado})"


class Giro90Aleatorio:
    """Gira la imagen 0, 90, 180 o 270 grados, los cuatro con la misma probabilidad.

    Una lechuga recolectada no tiene "arriba": cualquier giro es una foto posible. Se aplica a
    imágenes ya cuadradas, así que la forma no cambia.
    """

    def __call__(self, imagen: torch.Tensor) -> torch.Tensor:
        vueltas = int(torch.randint(0, 4, (1,)))
        return torch.rot90(imagen, vueltas, dims=(-2, -1))

    def __repr__(self) -> str:
        return "Giro90Aleatorio()"


def _normalizar() -> list:
    return [v2.ToDtype(torch.float32, scale=True), v2.Normalize(IMAGENET_MEDIA, IMAGENET_DESV)]


def transformacion_eval(resolucion: int) -> v2.Compose:
    """La transformación de valid, test y ``predecir``: cuadrar y normalizar, sin azar."""
    return v2.Compose([Cuadrar(resolucion), *_normalizar()])


def transformacion_train(resolucion: int, aumentos: Aumentos) -> v2.Compose:
    """La transformación de train: cuadrar, aumentos aleatorios y normalizar.

    En lenguaje llano, en este orden: se encaja en el cuadrado (como en evaluación); se toma un
    recorte aleatorio con entre ``escala_recorte`` del área y la proporción ``proporcion_recorte``
    (con [1, 1], cuadrado: no se deforma) y se lleva a ``resolucion``; volteos horizontal y
    vertical y giros de 90 grados si están activados; una rotación de hasta ``rotacion_grados``
    (las esquinas que quedan vacías se rellenan con el color medio); y, solo si son mayores que 0,
    brillo y contraste. A ``ColorJitter`` nunca se le pasan saturación ni tono.
    """
    pasos = [
        Cuadrar(resolucion),
        v2.RandomResizedCrop(
            resolucion,
            scale=aumentos.escala_recorte,
            ratio=aumentos.proporcion_recorte,
            antialias=True,
        ),
    ]
    if aumentos.volteo_horizontal:
        pasos.append(v2.RandomHorizontalFlip(p=0.5))
    if aumentos.volteo_vertical:
        pasos.append(v2.RandomVerticalFlip(p=0.5))
    if aumentos.giros_90:
        pasos.append(Giro90Aleatorio())
    if aumentos.rotacion_grados > 0:
        pasos.append(v2.RandomRotation(aumentos.rotacion_grados, fill=list(COLOR_RELLENO)))
    if aumentos.brillo > 0 or aumentos.contraste > 0:
        pasos.append(
            v2.ColorJitter(brightness=aumentos.brillo or None, contrast=aumentos.contraste or None)
        )
    return v2.Compose([*pasos, *_normalizar()])


class DatasetRecortes(Dataset):
    """Los recortes de una partición para el ``DataLoader``.

    Cada elemento es ``(imagen, objetivos, indice)``: la imagen ya transformada (tensor float
    (3, lado, lado)), las seis etiquetas codificadas (−1 = NO_VISIBLE) y la posición del recorte
    en las listas de entrada, para saber después de qué fila (y de qué fuente) era cada predicción.
    Las imágenes se leen con ``leer_imagen``, que admite rutas con tildes y eñes en Windows.
    """

    def __init__(self, rutas: list[Path], objetivos: np.ndarray, transformacion):
        if len(rutas) != len(objetivos):
            raise ValueError(
                f"hay {len(rutas)} rutas y {len(objetivos)} filas de objetivos: deben coincidir"
            )
        self.rutas = [Path(r) for r in rutas]
        self.objetivos = torch.as_tensor(np.asarray(objetivos), dtype=torch.int64)
        self.transformacion = transformacion

    def __len__(self) -> int:
        return len(self.rutas)

    def __getitem__(self, indice: int):
        imagen = a_tensor_rgb(leer_imagen(self.rutas[indice]))
        return self.transformacion(imagen), self.objetivos[indice], indice
