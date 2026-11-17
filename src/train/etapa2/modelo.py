"""Modelo de la etapa 2: un extractor de timm compartido y una cabeza lineal por grupo.

El extractor (EfficientNet-B0 preentrenada en ImageNet, sin su clasificador) convierte cada
recorte en un vector de características; cada grupo tiene su propia cabeza (dropout + capa lineal)
con tantas salidas como etiquetas tiene el grupo en ``schema.py`` (sin NO_VISIBLE).

PROVISIONAL (decisiones del tutor): una sola etiqueta por grupo (softmax) mientras no se decida
la multietiqueta. ``predecir`` nunca devuelve NO_VISIBLE ni la salida global: decidir cuándo un
grupo no se ve y agregar los seis grupos es cosa de la v2-03.

Solo ``crear_extractor_timm`` importa ``timm`` (dentro de la función): el resto, y las pruebas,
no lo necesitan.
"""

import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.data.schema import GROUPS
from src.train.etapa2.aumentos import (
    IMAGENET_DESV,
    IMAGENET_MEDIA,
    a_tensor_rgb,
    transformacion_eval,
)

# Firma de las funciones que crean el extractor: (arquitectura, preentrenado) -> (módulo, n).
CreadorExtractor = Callable[[str, bool], tuple[nn.Module, int]]


class ClasificadorEtapa2(nn.Module):
    """Extractor compartido y una cabeza ``Dropout + Linear`` por grupo.

    ``forward`` devuelve ``{grupo: logits}`` en el orden de ``GROUPS``; los logits de un grupo
    tienen forma (lote, número de etiquetas del grupo). Guarda lo necesario para reconstruirse y
    para predecir: grupos y etiquetas, resolución de entrada, dropout y arquitectura.
    """

    def __init__(self, extractor: nn.Module, n_caracteristicas: int, *, dropout: float,
                 resolucion: int, arquitectura: str, grupos: dict | None = None):  # fmt: skip
        super().__init__()
        self.extractor = extractor
        self.n_caracteristicas = n_caracteristicas
        self.grupos = {g: tuple(e) for g, e in (grupos or GROUPS).items()}
        self.dropout = dropout
        self.resolucion = resolucion
        self.arquitectura = arquitectura
        self.cabezas = nn.ModuleDict({
            grupo: nn.Sequential(nn.Dropout(dropout), nn.Linear(n_caracteristicas, len(etiquetas)))
            for grupo, etiquetas in self.grupos.items()
        })  # fmt: skip

    def forward(self, imagenes: torch.Tensor) -> dict[str, torch.Tensor]:
        caracteristicas = self.extractor(imagenes)
        return {grupo: cabeza(caracteristicas) for grupo, cabeza in self.cabezas.items()}


def crear_extractor_timm(arquitectura: str, preentrenado: bool) -> tuple[nn.Module, int]:
    """El modelo de timm sin su clasificador, y cuántas características da.

    En lenguaje llano: ``num_classes=0`` quita la última capa de ImageNet (sus 1000 clases) y deja
    el vector de características tras la media global; ``drop_rate=0`` porque el dropout ya lo pone
    cada cabeza. Con ``preentrenado`` timm descarga los pesos de ImageNet (de Hugging Face) la
    primera vez. Se comprueba que esos pesos esperan la normalización de ImageNet que usan las
    transformaciones.

    Errores: ``ImportError`` si timm no está instalado; ``ValueError`` si la normalización de los
    pesos no es la de ImageNet.
    """
    try:
        import timm
    except ImportError as error:
        raise ImportError(
            "No está instalado timm. Instálalo con `pip install -c <constraints> -r "
            "requirements-train.txt` (ver notebooks/03_entrenamiento_etapa2.ipynb)"
        ) from error
    modelo = timm.create_model(arquitectura, pretrained=preentrenado, num_classes=0, drop_rate=0.0)
    configuracion = getattr(modelo, "pretrained_cfg", None) or {}
    media, desv = configuracion.get("mean"), configuracion.get("std")
    if media is not None and desv is not None:
        pares = zip((*media, *desv), (*IMAGENET_MEDIA, *IMAGENET_DESV), strict=True)
        iguales = all(math.isclose(a, b, abs_tol=1e-3) for a, b in pares)
        if not iguales:
            raise ValueError(
                f"{arquitectura}: sus pesos esperan otra normalización (media {tuple(media)}, "
                f"desviación {tuple(desv)}) y las transformaciones usan la de ImageNet"
            )
    return modelo, int(modelo.num_features)


def crear_modelo(
    arquitectura: str,
    preentrenado: bool,
    dropout: float,
    resolucion: int,
    crear_extractor: CreadorExtractor = crear_extractor_timm,
) -> ClasificadorEtapa2:
    """Crea el clasificador con el extractor que dé ``crear_extractor`` (timm por defecto)."""
    extractor, n = crear_extractor(arquitectura, preentrenado)
    return ClasificadorEtapa2(extractor, n, dropout=dropout, resolucion=resolucion,
                              arquitectura=arquitectura)  # fmt: skip


def guardar_pesos(ruta: str | Path, modelo: ClasificadorEtapa2, metadatos: dict) -> None:
    """Guarda los pesos y lo necesario para reconstruir el modelo (sin rutas ni datos personales).

    El archivo es un diccionario con ``estado`` (los pesos), ``arquitectura``, ``resolucion``,
    ``dropout``, ``n_caracteristicas``, ``grupos`` (las etiquetas de cada uno), la normalización
    (``media`` y ``desv``) y los ``metadatos`` que se pasen (época, criterio...). Todo son tipos
    simples, así que se puede leer con ``torch.load(..., weights_only=True)``.
    """
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "estado": modelo.state_dict(),
        "arquitectura": modelo.arquitectura,
        "resolucion": modelo.resolucion,
        "dropout": modelo.dropout,
        "n_caracteristicas": modelo.n_caracteristicas,
        "grupos": {g: list(e) for g, e in modelo.grupos.items()},
        "media": list(IMAGENET_MEDIA),
        "desv": list(IMAGENET_DESV),
        **metadatos,
    }, ruta)  # fmt: skip


def cargar_modelo(
    ruta: str | Path,
    dispositivo: str = "cpu",
    crear_extractor: CreadorExtractor = crear_extractor_timm,
) -> ClasificadorEtapa2:
    """Reconstruye un modelo guardado con ``guardar_pesos``, listo para ``predecir``.

    En lenguaje llano: crea la misma arquitectura sin descargar los pesos de ImageNet (se van a
    sustituir), le carga los pesos guardados, lo lleva a ``dispositivo`` y lo deja en modo
    evaluación. Es lo que usará la v2-03.

    Error: ``ValueError`` si el extractor no da las características con que se guardó.
    """
    datos = torch.load(Path(ruta), map_location=dispositivo, weights_only=True)
    extractor, n = crear_extractor(datos["arquitectura"], False)
    if n != datos["n_caracteristicas"]:
        raise ValueError(f"el extractor da {n} características y los pesos se guardaron con "
                         f"{datos['n_caracteristicas']}")  # fmt: skip
    modelo = ClasificadorEtapa2(
        extractor,
        n,
        dropout=datos["dropout"],
        resolucion=datos["resolucion"],
        arquitectura=datos["arquitectura"],
        grupos=datos["grupos"],
    )
    modelo.load_state_dict(datos["estado"])
    return modelo.to(dispositivo).eval()


@torch.no_grad()
def predecir(modelo: ClasificadorEtapa2, imagenes: list[np.ndarray], tamano_lote: int = 32
             ) -> list[dict[str, dict]]:  # fmt: skip
    """Etiqueta y probabilidades de cada grupo para cada imagen.

    En lenguaje llano: cada imagen (BGR ``uint8`` de OpenCV, como las de ``src/pipeline``, de
    cualquier tamaño) se encaja en el cuadrado de entrada y se normaliza igual que en valid; se
    pasan al modelo por lotes y, en cada grupo, las salidas se convierten en probabilidades
    (softmax) y la etiqueta es la más probable. Devuelve una lista, en el orden de entrada, de
    ``{grupo: {"etiqueta": str, "probabilidades": {etiqueta: float}}}``.

    Nunca devuelve NO_VISIBLE: decidir si un grupo se ve es cosa de la v2-03 (PENDIENTE). El
    modelo vuelve al modo (train o eval) en que estaba.
    """
    if not imagenes:
        return []
    transformacion = transformacion_eval(modelo.resolucion)
    dispositivo = next(modelo.parameters()).device
    estaba_entrenando = modelo.training
    modelo.eval()
    resultados = []
    try:
        for inicio in range(0, len(imagenes), tamano_lote):
            lote = torch.stack([transformacion(a_tensor_rgb(imagen))
                                for imagen in imagenes[inicio : inicio + tamano_lote]])  # fmt: skip
            salidas = modelo(lote.to(dispositivo))
            probabilidades = {g: torch.softmax(s.float(), dim=1).cpu() for g, s in salidas.items()}
            for i in range(lote.shape[0]):
                resultado = {}
                for grupo, etiquetas in modelo.grupos.items():
                    fila = probabilidades[grupo][i].tolist()
                    resultado[grupo] = {
                        "etiqueta": etiquetas[int(np.argmax(fila))],
                        "probabilidades": dict(zip(etiquetas, fila, strict=True)),
                    }
                resultados.append(resultado)
    finally:
        modelo.train(estaba_entrenando)
    return resultados
