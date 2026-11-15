"""Lectura de la hoja de captura (xlsx) de las fotos propias y traducción de sus textos.

Qué hace, en lenguaje llano: la hoja la rellena una persona a mano, así que un mismo estado puede
venir escrito de varias formas ("Pocas hojas rasgada", "POCAS HOJAS RASGADAS") y los recuentos
pueden ser aproximados ("~10", "1-2", "Todas"). Este módulo lee la hoja con openpyxl y ofrece
funciones pequeñas que convierten esos textos en códigos y números. No decide ninguna etiqueta:
eso lo hace ``src/data/propio.py`` con las reglas del YAML.
"""

import numbers
import re
import unicodedata
from pathlib import Path

from src.data.schema import is_blank

# La fila de cabecera se busca entre las primeras filas: encima puede haber un título.
FILAS_CABECERA = 20

# Celda de un estado "no hecho": solo guiones, al menos dos (también los largos que pone la
# autocorrección: ‐ ‑ ‒ – — ―). Un guion suelto no cuenta: podría ser un "cero" o un descuido.
_GUIONES = re.compile(r"[-‐-―]{2,}")
_ENTERO = re.compile(r"[0-9]+")
_APROXIMADO = re.compile(r"~\s*([0-9]+)")
_RANGO = re.compile(r"([0-9]+)\s*-\s*([0-9]+)")
_PALABRA = re.compile(r"[a-z0-9]+")


def normalizar(texto) -> str:
    """Forma "canónica" de un texto para compararlo sin que importen detalles de escritura.

    En lenguaje llano: pasa a minúsculas, quita tildes y eñes (``ñ`` → ``n``), cambia los signos
    de puntuación por espacios, junta los espacios repetidos y quita una ``s`` final a cada palabra
    de más de tres letras (plural → singular: "hojas rasgadas" → "hoja rasgada"). Así "Pocas
    hojas rasgada" y "POCAS  HOJAS RASGADAS" quedan iguales. Las palabras cortas ("las", "sus")
    no se tocan.
    """
    descompuesto = unicodedata.normalize("NFKD", str(texto))
    sin_tildes = "".join(c for c in descompuesto if not unicodedata.combining(c)).lower()
    palabras = _PALABRA.findall(sin_tildes)
    return " ".join(p[:-1] if len(p) > 3 and p.endswith("s") else p for p in palabras)


def texto_celda(valor) -> str:
    """Texto de una celda tal como se leería en la hoja (``""`` si está vacía).

    En lenguaje llano: Excel guarda a veces los enteros como decimales (``5.0``); aquí se
    escriben sin decimales (``"5"``). El resto se convierte a texto y se quitan los espacios de
    los extremos.
    """
    if is_blank(valor):
        return ""
    if isinstance(valor, numbers.Real) and not isinstance(valor, bool):
        if float(valor).is_integer():
            return str(int(valor))
    return str(valor).strip()


def traducir_estado(valor, alias: dict[str, str]) -> str | None:
    """Código del estado (``"5"``, ``"24h"``...) que corresponde al texto de una celda.

    ``alias`` es ``{texto normalizado: código}`` (lo construye ``validar_reglas`` a partir de
    ``estados`` del YAML e incluye cada código tal cual). Devuelve ``None`` si la celda está vacía
    o el texto no corresponde a ningún estado.
    """
    if is_blank(valor):
        return None
    return alias.get(normalizar(texto_celda(valor)))


def leer_recuento(valor, palabras: dict[str, int]) -> int | None:
    """Número de hojas de una celda de recuento; ``None`` si está vacía (recuento desconocido).

    En lenguaje llano, acepta: un entero (``3``, o ``3.0`` si Excel lo guardó así), ``"~10"``
    (aproximado: cuenta como 10), ``"1-2"`` (un rango: cuenta el máximo, 2) y las palabras de
    ``palabras`` (ya normalizadas, p. ej. ``{"toda": 5}`` para "Todas"). Cualquier otra cosa
    (texto, negativo, decimal) lanza ``ValueError``: mejor parar que inventar un número.
    """
    if is_blank(valor):
        return None
    if isinstance(valor, numbers.Real) and not isinstance(valor, bool):
        if float(valor).is_integer() and valor >= 0:
            return int(valor)
    elif isinstance(valor, str):
        texto = valor.strip()
        if _ENTERO.fullmatch(texto):
            return int(texto)
        if coincidencia := _APROXIMADO.fullmatch(texto):
            return int(coincidencia.group(1))
        if coincidencia := _RANGO.fullmatch(texto):
            return max(int(coincidencia.group(1)), int(coincidencia.group(2)))
        if normalizar(texto) in palabras:
            return palabras[normalizar(texto)]
    raise ValueError(
        f"recuento '{valor}' no reconocido (usa un entero, ~N, N-M, una palabra de "
        "recuentos.palabras del YAML o déjalo vacío si no se sabe)"
    )


def leer_se_ve(valor) -> bool | None:
    """Respuesta de la columna "¿Se ve el daño en las fotos?": ``False`` (no), ``True`` (sí) o
    ``None`` (vacía). Cualquier otro texto lanza ``ValueError``."""
    if is_blank(valor):
        return None
    clave = normalizar(texto_celda(valor))
    if clave == "no":
        return False
    if clave == "si":
        return True
    raise ValueError(f"valor '{valor}' no reconocido (usa no, sí o vacío)")


def es_no_hecha(celdas) -> bool:
    """True si alguna celda de la fila tiene solo guiones (``----``): ese estado no se hizo."""
    return any(isinstance(c, str) and _GUIONES.fullmatch(c.strip()) for c in celdas)


def _leer_celdas(ruta: Path, nombre_hoja: str) -> list[list]:
    """Valores de todas las celdas de una hoja, fila a fila, con las combinadas ya rellenas.

    En lenguaje llano: en Excel, al combinar celdas (p. ej. la lechuga escrita una vez para
    varias filas) solo la primera guarda el valor y las demás quedan vacías. Aquí se copia ese
    valor a todas las celdas de la combinación, que es lo que ve quien mira la hoja.
    """
    try:
        import openpyxl
    except ImportError as error:  # dependencia de requirements.txt
        raise ImportError(
            "Falta openpyxl para leer la hoja de captura: pip install openpyxl"
        ) from error

    # Sin read_only: ese modo no informa de las celdas combinadas. data_only: el valor
    # calculado de las fórmulas, no la fórmula.
    libro = openpyxl.load_workbook(ruta, data_only=True)
    try:
        if nombre_hoja not in libro.sheetnames:
            raise ValueError(
                f"{ruta.name}: no tiene la hoja '{nombre_hoja}' "
                f"(hojas: {', '.join(libro.sheetnames)})"
            )
        hoja = libro[nombre_hoja]
        celdas = [list(fila) for fila in hoja.iter_rows(values_only=True)]
        for rango in hoja.merged_cells.ranges:
            valor = hoja.cell(rango.min_row, rango.min_col).value
            for fila in range(rango.min_row - 1, min(rango.max_row, len(celdas))):
                for columna in range(rango.min_col - 1, min(rango.max_col, len(celdas[fila]))):
                    celdas[fila][columna] = valor
    finally:
        libro.close()
    return celdas


def leer_hoja(ruta: str | Path, nombre_hoja: str, columnas: dict[str, str]) -> list[dict]:
    """Lee las filas de la hoja de captura como ``{clave interna: valor de la celda}``.

    En lenguaje llano: ``columnas`` dice qué cabecera de la hoja corresponde a cada clave interna
    (``{"lechuga": "Lechuga", "estado": "Qué le has hecho (estado)", ...}``). La fila de cabecera
    se busca entre las ``FILAS_CABECERA`` primeras y las cabeceras se comparan normalizadas (sin
    mayúsculas, tildes ni espacios de más). Las columnas que no están en ``columnas`` (Fecha,
    Hora...) no se devuelven, pero sí cuentan para detectar guiones.

    Cada fila devuelta lleva además ``"fila"`` (su número en Excel, contando desde 1) y
    ``"no_hecha"`` (``True`` si alguna celda tiene solo guiones). Las filas vacías del todo se
    saltan. Los valores se devuelven tal cual (texto, número, fecha o ``None``): traducirlos es
    cosa de quien llama.

    Errores (``ValueError``): falta la hoja, falta alguna cabecera (las nombra) o una cabecera
    está repetida. ``ImportError`` si falta openpyxl.
    """
    ruta = Path(ruta)
    celdas = _leer_celdas(ruta, nombre_hoja)
    buscadas = {clave: normalizar(cabecera) for clave, cabecera in columnas.items()}

    cabecera, mejor = None, {}
    for numero, fila in enumerate(celdas[:FILAS_CABECERA]):
        textos = [normalizar(c) if not is_blank(c) else "" for c in fila]
        posiciones = {clave: textos.index(b) for clave, b in buscadas.items() if b in textos}
        if len(posiciones) == len(buscadas):
            cabecera = numero
            repetidas = [columnas[clave] for clave, b in buscadas.items() if textos.count(b) > 1]
            if repetidas:
                raise ValueError(
                    f"{ruta.name}, hoja '{nombre_hoja}': cabecera repetida en la fila "
                    f"{numero + 1}: {', '.join(repetidas)}"
                )
            break
        if len(posiciones) > len(mejor):
            mejor = posiciones
    if cabecera is None:
        faltan = [columnas[clave] for clave in buscadas if clave not in mejor]
        raise ValueError(
            f"{ruta.name}, hoja '{nombre_hoja}': no encuentro la fila de cabecera en las "
            f"{FILAS_CABECERA} primeras filas; faltan las columnas: {', '.join(faltan)} "
            "(se comparan sin mayúsculas ni tildes)"
        )

    filas = []
    for numero in range(cabecera + 1, len(celdas)):
        fila = celdas[numero]
        if all(is_blank(c) for c in fila):
            continue
        registro = {"fila": numero + 1, "no_hecha": es_no_hecha(fila)}
        registro.update({clave: fila[i] for clave, i in posiciones.items()})
        filas.append(registro)
    return filas
