"""Lectura de los archivos de configuración de los experimentos (YAML)."""

from pathlib import Path

import yaml


def load_config(ruta: str | Path) -> dict:
    """Lee un archivo YAML y devuelve su contenido como diccionario.

    En lenguaje llano: abre el archivo de configuración (por ejemplo
    ``configs/v0.yaml``), lo convierte en un diccionario de Python y lo
    devuelve, para que el resto del código lea los parámetros de ahí en lugar
    de tenerlos escritos dentro del código.

    Errores:
        FileNotFoundError: si el archivo no existe (el mensaje incluye la ruta).
        ValueError: si el archivo está vacío o su contenido no es un diccionario.
        yaml.YAMLError: si el YAML está mal escrito (indica línea y columna).
    """
    ruta = Path(ruta)
    if not ruta.is_file():
        raise FileNotFoundError(f"No se encuentra el archivo de configuración: {ruta}")

    # utf-8 explícito: en Windows el valor por defecto (cp1252) rompería las tildes.
    with ruta.open(encoding="utf-8") as archivo:
        contenido = yaml.safe_load(archivo)

    if not isinstance(contenido, dict):
        raise ValueError(
            f"La configuración {ruta} debe ser un diccionario YAML (clave: valor), "
            f"pero contiene: {type(contenido).__name__}"
        )
    return contenido
