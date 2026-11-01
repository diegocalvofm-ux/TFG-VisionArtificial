"""Fixtures de las pruebas de src/train: configuración y datasets COCO sintéticos (sin rfdetr)."""

import copy
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

# Configuración válida mínima, con la misma forma que configs/train/etapa1_rfdetr_n.yaml.
CONFIG_VALIDA = {
    "modelo": {"clase": "RFDETRNano", "resolucion": 384, "clases": ["lechuga"]},
    "entrenamiento": {
        "epocas": 30,
        "batch_size": 8,
        "grad_accum_steps": 2,
        "lr": 1.0e-4,
        "lr_encoder": 1.5e-4,
        "amp_dtype": "auto",
        "num_workers": 2,
        "early_stopping": False,
        "checkpoint_interval": 10,
        "tensorboard": False,
        "dispositivo": "auto",
    },
    "semilla": 42,
    "datos": {
        "dataset": "data/processed/etapa1_v1",
        "repeticiones_svodk": 1,
        "prefijo_svodk": "svodk_",
        "trabajo": "data/processed/trabajo",
    },
    "salida": {"carpeta": "runs"},
    "smoke": {"epocas": 1, "imagenes_train": 4},
}

POLIGONO = [[2, 2, 10, 2, 10, 8, 2, 8]]


def _poner(config: dict, ruta: str, valor) -> None:
    """Pone ``valor`` en la clave ``a.b`` de un diccionario anidado."""
    *padres, hoja = ruta.split(".")
    destino = config
    for clave in padres:
        destino = destino[clave]
    destino[hoja] = valor


def _quitar(config: dict, ruta: str) -> None:
    *padres, hoja = ruta.split(".")
    destino = config
    for clave in padres:
        destino = destino[clave]
    del destino[hoja]


@pytest.fixture
def config_valida() -> dict:
    """Una copia de la configuración válida (se puede modificar sin afectar a otras pruebas)."""
    return copy.deepcopy(CONFIG_VALIDA)


@pytest.fixture
def escribir_config(tmp_path):
    """Devuelve una función que escribe un YAML de entrenamiento y devuelve su ruta.

    ``cambios`` es ``{"entrenamiento.epocas": 5}`` (clave con puntos -> valor) y ``quitar`` una
    lista de claves con puntos que se borran del YAML.
    """

    def fabrica(cambios: dict | None = None, quitar: tuple[str, ...] = (), nombre="config.yaml"):
        config = copy.deepcopy(CONFIG_VALIDA)
        for ruta, valor in (cambios or {}).items():
            _poner(config, ruta, valor)
        for ruta in quitar:
            _quitar(config, ruta)
        destino = tmp_path / nombre
        destino.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
        return destino

    return fabrica


def _jpeg(valor: int) -> bytes:
    """JPEG de 20x20 píxeles de un solo color. Nunca es una imagen real."""
    correcto, datos = cv2.imencode(".jpg", np.full((20, 20, 3), valor % 256, dtype=np.uint8))
    assert correcto
    return datos.tobytes()


def _escribir_carpeta(carpeta: Path, nombres: list[str]) -> None:
    """Escribe las imágenes y un ``_annotations.coco.json`` con una lechuga por imagen."""
    carpeta.mkdir(parents=True, exist_ok=True)
    imagenes, anotaciones = [], []
    for i, nombre in enumerate(nombres, start=1):
        (carpeta / nombre).write_bytes(_jpeg(i * 11))
        imagenes.append({"id": i, "file_name": nombre, "width": 20, "height": 20})
        anotaciones.append(
            {
                "id": i,
                "image_id": i,
                "category_id": 1,
                "bbox": [2, 2, 8, 6],
                "area": 48,
                "iscrowd": 0,
                "segmentation": POLIGONO,
            }
        )
    contenido = {
        "images": imagenes,
        "annotations": anotaciones,
        "categories": [{"id": 1, "name": "lechuga"}],
    }
    (carpeta / "_annotations.coco.json").write_text(json.dumps(contenido), encoding="utf-8")


@pytest.fixture
def dataset_sintetico(tmp_path):
    """Devuelve una función que crea un dataset estilo etapa1_v1 y devuelve su carpeta.

    train: ``n_svodk`` imágenes ``svodk_NNNN.jpg`` y ``n_giovi`` ``giovi_NNNN.jpg``; valid:
    ``n_valid`` imágenes de svodk; test vacío. Lleva un ``informe.json`` como el del importador.
    """

    def fabrica(destino: Path | None = None, n_svodk=3, n_giovi=5, n_valid=2) -> Path:
        destino = destino or tmp_path / "dataset"
        _escribir_carpeta(
            destino / "train",
            [f"svodk_{i:04d}.jpg" for i in range(1, n_svodk + 1)]
            + [f"giovi_{i:04d}.jpg" for i in range(1, n_giovi + 1)],
        )
        _escribir_carpeta(
            destino / "valid", [f"svodk_{i:04d}.jpg" for i in range(100, 100 + n_valid)]
        )
        _escribir_carpeta(destino / "test", [])
        informe = {"semilla": 42, "totales": {"train": n_svodk + n_giovi, "valid": n_valid}}
        (destino / "informe.json").write_text(json.dumps(informe), encoding="utf-8")
        return destino

    return fabrica
