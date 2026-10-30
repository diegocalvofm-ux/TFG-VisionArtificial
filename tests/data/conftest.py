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
