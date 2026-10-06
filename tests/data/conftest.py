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
_VISTAS = ("cenital", "lateral", "base")


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
                }
                filas.append(fila)
        return pd.DataFrame(filas)

    return fabrica
