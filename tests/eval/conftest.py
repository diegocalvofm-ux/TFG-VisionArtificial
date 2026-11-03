"""Fixtures de las pruebas de src/eval: conjunto COCO sintético, configuración y métodos falsos.

Nada usa modelos reales ni datos reales: tres imágenes grises de 120x100 píxeles, distinguibles por
su nivel de gris (100, 110 y 120), y detectores falsos que devuelven cajas fijas según ese nivel.

Cajas de referencia (x1, y1, x2, y2):
    a_ñ.png (gris 100): (10, 10, 50, 50) y (60, 10, 100, 50)
    b.png   (gris 110): (20, 20, 80, 80)
    c.png   (gris 120): (30, 30, 70, 70)
"""

import dataclasses
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.eval.config import cargar_config_localizacion
from src.eval.metodos import Metodo
from src.pipeline.types import Detection

RAIZ = Path(__file__).resolve().parents[2]
YAML_REAL = RAIZ / "configs" / "eval" / "localizacion.yaml"

ANCHO, ALTO = 120, 100

# nombre de archivo -> (gris de la imagen, cajas de referencia)
IMAGENES = {
    "a_ñ.png": (100, [(10, 10, 50, 50), (60, 10, 100, 50)]),
    "b.png": (110, [(20, 20, 80, 80)]),
    "c.png": (120, [(30, 30, 70, 70)]),
}

# Latencia simulada por imagen (ms), según el gris de la imagen.
MS_POR_GRIS = {100: 10.0, 110: 20.0, 120: 30.0}

# Predicciones (caja, confianza) del método con confianza, según el gris de la imagen.
#  a: dos aciertos, un duplicado (FP) y una caja de confianza baja bajo el umbral operativo.
#  b: nada.  c: una caja sin objeto.
PREDICCIONES_CON_CONFIANZA = {
    100: [((10, 10, 50, 50), 0.9), ((60, 10, 100, 50), 0.8), ((12, 12, 52, 52), 0.7),
          ((0, 60, 20, 80), 0.3)],  # fmt: skip
    110: [],
    120: [((0, 0, 10, 10), 0.9)],
}
# Predicciones del método sin confianza (línea base falsa).
PREDICCIONES_SIN_CONFIANZA = {
    100: [(10, 10, 50, 50), (60, 10, 100, 50)],
    110: [(20, 20, 80, 80)],
    120: [],
}


def escribir_conjunto(raiz: Path, carpeta: str = "conjunto", particion: str = "valid") -> Path:
    """Crea ``<raiz>/<carpeta>/<particion>/`` con imágenes grises y su JSON de anotaciones COCO."""
    destino = raiz / carpeta / particion
    destino.mkdir(parents=True, exist_ok=True)
    imagenes, anotaciones = [], []
    for id_imagen, (nombre, (gris, cajas)) in enumerate(IMAGENES.items(), start=1):
        imagen = np.full((ALTO, ANCHO, 3), gris, dtype=np.uint8)
        correcto, datos = cv2.imencode(".png", imagen)
        assert correcto
        datos.tofile(destino / nombre)
        imagenes.append({"id": id_imagen, "file_name": nombre, "width": ANCHO, "height": ALTO})
        for x1, y1, x2, y2 in cajas:
            ancho, alto = x2 - x1, y2 - y1
            anotaciones.append(
                {"id": len(anotaciones) + 1, "image_id": id_imagen, "category_id": 1,
                 "bbox": [x1, y1, ancho, alto], "area": ancho * alto, "iscrowd": 0}
            )  # fmt: skip
    coco = {"images": imagenes, "annotations": anotaciones,
            "categories": [{"id": 1, "name": "lechuga"}]}  # fmt: skip
    (destino / "_annotations.coco.json").write_text(
        json.dumps(coco, ensure_ascii=False), encoding="utf-8"
    )
    return destino


@pytest.fixture
def raiz_sintetica(tmp_path):
    """Una raíz de repositorio falsa con el conjunto sintético ya escrito."""
    escribir_conjunto(tmp_path)
    return tmp_path


@pytest.fixture
def config_real():
    """La configuración real de configs/eval/localizacion.yaml, ya validada."""
    return cargar_config_localizacion(YAML_REAL)


@pytest.fixture
def config_sintetica(config_real):
    """La configuración real adaptada al conjunto sintético y a una medición corta."""
    return dataclasses.replace(
        config_real,
        conjunto_carpeta="conjunto",
        repeticiones=2,
        calentamiento=1,
        ejemplos_numero=2,
    )


class RelojFalso:
    """Un reloj que solo avanza cuando se le dice. Sustituye a ``time.perf_counter``."""

    def __init__(self):
        self.segundos = 0.0

    def __call__(self) -> float:
        return self.segundos

    def avanzar(self, milisegundos: float) -> None:
        self.segundos += milisegundos / 1000


@pytest.fixture
def reloj():
    return RelojFalso()


@pytest.fixture
def fabricar_metodo(reloj):
    """Devuelve una función que crea un método simulado: cajas fijas y el reloj que avanza."""

    def fabrica(con_confianza: bool = True) -> Metodo:
        def predecir(imagen):
            gris = int(imagen[0, 0, 0])
            reloj.avanzar(MS_POR_GRIS[gris])
            if con_confianza:
                return [Detection(box=c, score=s) for c, s in PREDICCIONES_CON_CONFIANZA[gris]]
            return [Detection(box=c, score=1.0) for c in PREDICCIONES_SIN_CONFIANZA[gris]]

        return Metodo(
            tiene_confianza=con_confianza,
            predecir=predecir,
            tiempo_carga_s=0.5,
            umbral=0.5 if con_confianza else None,
            umbral_ap=0.05 if con_confianza else None,
            parametros={"detalle": "simulado"},
        )

    return fabrica


@pytest.fixture
def fabricas(fabricar_metodo):
    """Fábricas de los tres métodos del YAML, todas simuladas."""
    return {
        "owlvit": lambda: fabricar_metodo(True),
        "rfdetr": lambda: fabricar_metodo(True),
        "linea_base_hsv": lambda: fabricar_metodo(False),
    }


@pytest.fixture
def entorno_falso():
    """Entorno simulado (sin llamar a git ni importar torch)."""
    return lambda: {"python": "3.12.0", "commit": "abc1234", "dirty": False, "device": "cpu"}
