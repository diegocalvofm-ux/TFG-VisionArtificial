"""Fixtures de las pruebas de src/train/etapa2: configuración, recortes sintéticos y un extractor
diminuto que sustituye a EfficientNet (sin timm, sin descargar pesos y sin GPU).

El dataset sintético imita a ``data/processed/etapa2_v1``: un ``recortes.csv`` válido para
``validar_recortes`` y una imagen JPEG diminuta por recorte en ``imagenes/``.

- ``propio``: 6 lechugas x 2 vistas = 12 recortes, todos en train (como en el piloto), con
  etiquetas en los seis grupos y algún NO_VISIBLE.
- ``yssp`` y ``phs``: 8 recortes en train, 4 en valid y 4 en test cada una, con etiqueta solo en
  ``daño_bio`` (los otros cinco grupos en NO_VISIBLE).
"""

import copy
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import yaml

from src.data.dataset_etapa2 import COLUMNAS
from src.data.schema import GROUPS, NO_VISIBLE

# Configuración válida mínima, con la misma forma que configs/train/etapa2_effb0.yaml pero con
# valores pequeños para que el entrenamiento de las pruebas dure menos de un segundo en CPU.
CONFIG_VALIDA = {
    "modelo": {
        "arquitectura": "efficientnet_b0",
        "preentrenado": False,
        "resolucion": 32,
        "dropout": 0.2,
    },
    "entrenamiento": {
        "epocas": 2,
        "batch_size": 4,
        "lr_extractor": 1.0e-3,
        "lr_cabezas": 1.0e-2,
        "decaimiento_pesos": 1.0e-2,
        "epocas_calentamiento": 1,
        "factor_lr_final": 0.01,
        "precision_mixta": True,  # en CPU no se usa: así se prueba que se ignora
        "num_workers": 0,
        "dispositivo": "cpu",
    },
    "perdida": {"pesos_clase": "inv_raiz"},
    "aumentos": {
        "escala_recorte": [0.9, 1.0],
        "proporcion_recorte": [1.0, 1.0],
        "volteo_horizontal": True,
        "volteo_vertical": True,
        "giros_90": True,
        "rotacion_grados": 10,
        "brillo": 0.0,
        "contraste": 0.0,
    },
    "semilla": 42,
    "datos": {
        "dataset": "data/processed/etapa2_v1",
        "fuente_propia": "propio",
        "sobremuestreo_propio": 1,
        "reparto_propio": None,
    },
    "salida": {"carpeta": "runs"},
    "smoke": {"epocas": 1, "recortes_train": 6, "recortes_eval": 4},
}

# Etiquetas de los recortes propios: se recorren en ciclo (recorte i -> elemento i % len).
CICLOS_PROPIO = {
    "corte": ("CUT_OK", "STEM_TOO_LONG", NO_VISIBLE),
    "recorte": ("TRIM_OK", "UNDER_TRIMMED", "OVER_TRIMMED"),
    "hojas_ext": ("OUTER_LEAVES_OK", "OUTER_DAMAGE_LIGHT"),
    "color": ("COLOR_FRESH_OK", "COLOR_FRESH_OK", "YELLOW_BROWN_LIGHT"),
    "daño_fisico": ("PHYSICAL_OK", "TORN_LEAF", NO_VISIBLE),
    "daño_bio": ("BIO_OK",),
}
CICLOS_BIO = {
    "yssp": ("BIO_OK", "DISEASE_VISIBLE"),
    "phs": ("BIO_OK", "ROT_DECAY", "DISEASE_VISIBLE"),
}
N_PROPIO = 12
N_PUBLICA = {"train": 8, "valid": 4, "test": 4}


def poner(config: dict, ruta: str, valor) -> None:
    """Pone ``valor`` en la clave ``a.b`` de un diccionario anidado."""
    *padres, hoja = ruta.split(".")
    destino = config
    for clave in padres:
        destino = destino[clave]
    destino[hoja] = valor


def quitar(config: dict, ruta: str) -> None:
    *padres, hoja = ruta.split(".")
    destino = config
    for clave in padres:
        destino = destino[clave]
    del destino[hoja]


@pytest.fixture
def config_valida() -> dict:
    return copy.deepcopy(CONFIG_VALIDA)


def escribir_config_en(carpeta: Path, cambios: dict | None = None, sin: tuple[str, ...] = (),
                       nombre: str = "config.yaml") -> Path:  # fmt: skip
    """Escribe un YAML de entrenamiento con ``cambios`` (``clave.con.puntos`` -> valor) y sin
    las claves de ``sin``."""
    config = copy.deepcopy(CONFIG_VALIDA)
    for ruta, valor in (cambios or {}).items():
        poner(config, ruta, valor)
    for ruta in sin:
        quitar(config, ruta)
    destino = Path(carpeta) / nombre
    destino.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    return destino


@pytest.fixture
def escribir_config(tmp_path):
    """Función que escribe un YAML de entrenamiento en ``tmp_path`` y devuelve su ruta."""

    def fabrica(cambios: dict | None = None, sin: tuple[str, ...] = (), nombre="config.yaml"):
        return escribir_config_en(tmp_path, cambios, sin, nombre)

    return fabrica


def _jpeg(valor: int, ancho: int = 24, alto: int = 16) -> bytes:
    """JPEG diminuto (no cuadrado) de colores sintéticos. Nunca es una imagen real."""
    imagen = np.zeros((alto, ancho, 3), dtype=np.uint8)
    imagen[:, :, 0] = (valor * 37) % 256
    imagen[:, :, 1] = (valor * 91) % 256
    imagen[: alto // 2, :, 2] = 200
    correcto, datos = cv2.imencode(".jpg", imagen)
    assert correcto
    return datos.tobytes()


def _fila(fuente: str, numero: int, particion: str, lechuga: str, etiquetas: dict,
          vista: str = "") -> dict:  # fmt: skip
    recorte_id = f"{fuente}_{numero:06d}"
    return {
        "recorte_id": recorte_id,
        "fuente": fuente,
        "licencia": "CC BY 4.0",
        "imagen_origen": f"original_{recorte_id}.jpg",
        "caja_x": "0",
        "caja_y": "0",
        "caja_w": "24",
        "caja_h": "16",
        "origen_caja": "rfdetr" if fuente == "propio" else "anotacion",
        "lechuga_id": lechuga,
        "vista": vista,
        "particion": particion,
        **etiquetas,
        "etiqueta_original": "sintetica",
        "salida_global": "REVIEW",
        "dhash": "0123456789abcdef",
    }


def filas_sinteticas() -> list[dict]:
    """Las filas de ``recortes.csv`` del dataset sintético (ver el docstring del módulo)."""
    filas = []
    for i in range(N_PROPIO):
        etiquetas = {g: ciclo[i % len(ciclo)] for g, ciclo in CICLOS_PROPIO.items()}
        vista = ("cenital", "base")[i % 2]
        filas.append(_fila("propio", i + 1, "train", f"L{i // 2 + 1}", etiquetas, vista))
    for fuente, ciclo in CICLOS_BIO.items():
        numero = 0
        for particion, cantidad in N_PUBLICA.items():
            for _ in range(cantidad):
                numero += 1
                etiquetas = dict.fromkeys(GROUPS, NO_VISIBLE)
                etiquetas["daño_bio"] = ciclo[numero % len(ciclo)]
                filas.append(_fila(fuente, numero, particion, f"{fuente}_img{numero}", etiquetas))
    return filas


def escribir_dataset(destino: Path, filas: list[dict] | None = None) -> Path:
    """Escribe ``recortes.csv``, ``imagenes/`` e ``informe.json`` en ``destino``."""
    filas = filas_sinteticas() if filas is None else filas
    (destino / "imagenes").mkdir(parents=True, exist_ok=True)
    with (destino / "recortes.csv").open("w", encoding="utf-8", newline="") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=list(COLUMNAS), lineterminator="\n")
        escritor.writeheader()
        escritor.writerows(filas)
    for posicion, fila in enumerate(filas):
        (destino / "imagenes" / f"{fila['recorte_id']}.jpg").write_bytes(_jpeg(posicion))
    informe = {"semilla": 42, "totales": {"train": 28, "valid": 8, "test": 8}}
    (destino / "informe.json").write_text(json.dumps(informe), encoding="utf-8")
    return destino


@pytest.fixture
def dataset_sintetico(tmp_path):
    """Función que crea el dataset sintético (por defecto en ``tmp_path/dataset``)."""

    def fabrica(destino: Path | None = None, filas: list[dict] | None = None) -> Path:
        return escribir_dataset(destino or tmp_path / "dataset", filas)

    return fabrica


# --- Extractor diminuto -----------------------------------------------------------------------


def crear_extractor_diminuto(arquitectura: str, preentrenado: bool):
    """Sustituye a ``crear_extractor_timm``: una convolución y una media, 4 características.

    Devuelve ``(extractor, numero_de_caracteristicas)`` como la función real. Se importa torch
    aquí dentro para que las pruebas que no lo usan no lo necesiten.
    """
    from torch import nn

    extractor = nn.Sequential(
        nn.Conv2d(3, 4, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
    )
    return extractor, 4


@pytest.fixture
def extractor_diminuto():
    """La función ``crear_extractor_diminuto`` (para inyectarla en lugar de timm)."""
    return crear_extractor_diminuto


@pytest.fixture(scope="session")
def fabricas():
    """Las funciones de este archivo, para fixtures con alcance de módulo (no usan tmp_path)."""
    return SimpleNamespace(
        escribir_dataset=escribir_dataset,
        filas=filas_sinteticas,
        escribir_config=escribir_config_en,
        extractor=crear_extractor_diminuto,
    )
