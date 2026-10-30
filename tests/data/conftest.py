"""Fixtures compartidas por las pruebas de src/data (datos sintéticos, sin imágenes)."""

import pandas as pd
import pytest

# Valores "OK" de cada grupo, escritos a mano a propósito: así la fábrica no depende
# del código que se está probando.
_VALORES_OK = {
    "corte": "CUT_OK",
    "recorte": "TRIM_OK",
    "hojas_ext": "OUTER_LEAVES_OK",
    "color": "COLOR_FRESH_OK",
    "daño_fisico": "PHYSICAL_OK",
    "daño_bio": "BIO_OK",
}
_VISTAS = ("cenital", "lateral_a", "lateral_b", "base")
# Columnas de contexto de captura: opcionales, así que en la tabla mínima válida van vacías.
_CONTEXTO_VACIO = {
    "variedad": "",
    "estado_seq": "",
    "horas_desde_corte": "",
    "conservacion": "",
    "temperatura_c": "",
    "intervencion": "",
    "dispositivo": "",
    "archivo_original": "",
}


@pytest.fixture
def metadata_valida():
    """Devuelve una función que fabrica un DataFrame de metadatos válido.

    Uso: ``df = metadata_valida(n_lechugas=10, fotos_por_lechuga=3)``. Cada lechuga
    (``L000``, ``L001``...) tiene varias fotos con ``image_id`` únicos (``img_0000``...).
    """

    def fabrica(n_lechugas: int = 3, fotos_por_lechuga: int = 2) -> pd.DataFrame:
        filas = []
        for i in range(n_lechugas):
            for j in range(fotos_por_lechuga):
                fila = {
                    "image_id": f"img_{len(filas):04d}",
                    "fuente": "propia",
                    "licencia": "propia",
                    "etiqueta_original": "",
                    "lechuga_id": f"L{i:03d}",
                    "vista": _VISTAS[j % len(_VISTAS)],
                    "sesion": "s1",
                    **_VALORES_OK,
                    "salida_global": "",
                    **_CONTEXTO_VACIO,
                }
                filas.append(fila)
        df = pd.DataFrame(filas)
        # Tipo ``object`` en las columnas de contexto: con el tipo texto de pandas 3 no se podría
        # asignar un número o un booleano a una celda, y las pruebas necesitan probar esos casos.
        return df.astype(dict.fromkeys(_CONTEXTO_VACIO, object))

    return fabrica


# --- ZIP COCO sintéticos (estilo Roboflow) para las pruebas del importador -----------------


def _jpeg_diminuto(semilla: int) -> bytes:
    """JPEG de 20x20 píxeles de un solo color (distinto según la semilla). Nunca es real."""
    import cv2
    import numpy as np

    imagen = np.full((20, 20, 3), (semilla * 7) % 256, dtype=np.uint8)
    correcto, datos = cv2.imencode(".jpg", imagen)
    assert correcto
    return datos.tobytes()


def _escribir_zip_coco(ruta, carpetas: dict, categorias: tuple[str, ...]) -> None:
    """Escribe un ZIP con carpetas train/valid/test y un _annotations.coco.json por carpeta.

    ``carpetas`` es ``{carpeta: [{"nombre": "a.jpg", "anotaciones": [(clase, poligonos)]}]}``.
    ``categorias`` se numera desde 0; como en Roboflow, la primera suele ser una categoría
    "padre" sin anotaciones.
    """
    import json
    import zipfile

    with zipfile.ZipFile(ruta, "w") as zf:
        for carpeta, imagenes in carpetas.items():
            coco = {
                "info": {},
                "licenses": [],
                "categories": [{"id": i, "name": n} for i, n in enumerate(categorias)],
                "images": [],
                "annotations": [],
            }
            for i, imagen in enumerate(imagenes):
                coco["images"].append(
                    {"id": i, "file_name": imagen["nombre"], "width": 20, "height": 20}
                )
                for clase, poligonos in imagen.get("anotaciones", []):
                    coco["annotations"].append(
                        {
                            "id": len(coco["annotations"]),
                            "image_id": i,
                            "category_id": categorias.index(clase),
                            "bbox": [0, 0, 1, 1],  # a propósito distinta de la real
                            "iscrowd": 0,
                            "area": 1,
                            "segmentation": poligonos,
                        }
                    )
                zf.writestr(f"{carpeta}/{imagen['nombre']}", _jpeg_diminuto(i + len(carpeta)))
            zf.writestr(f"{carpeta}/_annotations.coco.json", json.dumps(coco, ensure_ascii=False))


@pytest.fixture
def zip_coco_sintetico(tmp_path):
    """Devuelve una función que crea un ZIP COCO mínimo y devuelve su ruta."""

    def fabrica(
        carpetas: dict, categorias: tuple[str, ...] = ("padre", "lechuga", "kern"), nombre="x.zip"
    ):
        ruta = tmp_path / "zips" / nombre
        ruta.parent.mkdir(parents=True, exist_ok=True)
        _escribir_zip_coco(ruta, carpetas, categorias)
        return ruta

    return fabrica


@pytest.fixture
def fuente_sintetica(tmp_path):
    """Devuelve una función que crea ZIP + YAML de una fuente bajo ``tmp_path`` (la "raíz").

    Devuelve la ruta del YAML. El ``zip`` del YAML es relativo a ``tmp_path``, como en el repo.
    """
    import yaml

    def fabrica(
        nombre: str,
        carpetas: dict,
        clases: dict,
        *,
        categorias: tuple[str, ...] = ("padre", "lechuga", "kern"),
        papel: str = "train_only",
        fracciones: dict | None = None,
        grupo_regex: str | None = None,
        particion_temporal: dict | None = None,
        excluir: tuple[str, ...] = (),
        licencia: str = "MIT",
    ):
        ruta_zip = tmp_path / "zips" / f"{nombre}.zip"
        ruta_zip.parent.mkdir(parents=True, exist_ok=True)
        _escribir_zip_coco(ruta_zip, carpetas, categorias)
        config = {
            "fuente": nombre,
            "licencia": licencia,
            "zip": f"zips/{nombre}.zip",
            "clases": clases,
            "papel": papel,
        }
        if fracciones is not None:
            config["fracciones"] = fracciones
        if grupo_regex is not None:
            config["grupo_regex"] = grupo_regex
        if particion_temporal is not None:
            config["particion_temporal"] = particion_temporal
        if excluir:
            config["excluir_imagenes"] = list(excluir)
        ruta_yaml = tmp_path / "configs" / "datasets" / f"{nombre}.yaml"
        ruta_yaml.parent.mkdir(parents=True, exist_ok=True)
        ruta_yaml.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
        return ruta_yaml

    return fabrica


@pytest.fixture
def etapa_sintetica(tmp_path):
    """Devuelve una función que escribe ``configs/datasets/etapa1_v1.yaml`` bajo ``tmp_path``."""
    import yaml

    def fabrica(fuentes: list[str], semilla: int = 42, salida: str = "data/processed/etapa1_v1"):
        ruta = tmp_path / "configs" / "datasets" / "etapa1_v1.yaml"
        ruta.parent.mkdir(parents=True, exist_ok=True)
        contenido = {"fuentes": fuentes, "semilla": semilla, "salida": salida}
        ruta.write_text(yaml.safe_dump(contenido), encoding="utf-8")
        return ruta

    return fabrica
