"""Esquema de los metadatos: grupos de calidad, valores válidos y validación.

Este módulo es la única fuente de verdad de los seis grupos y de sus valores. Otros
módulos (p. ej. la tabla de salida global) deben comprobarse contra lo definido aquí.
"""

import math
import numbers
import re

import pandas as pd

# Valor que se usa cuando un grupo no se puede ver en la foto. Vale en cualquier grupo.
NO_VISIBLE = "NO_VISIBLE"

# Grupo -> valores permitidos. No incluye NO_VISIBLE (es válido en todos los grupos).
GROUPS: dict[str, tuple[str, ...]] = {
    "corte": ("CUT_OK", "STEM_TOO_LONG", "UNCLEAN_CUT", "ROOT_REMAINS"),
    "recorte": ("TRIM_OK", "UNDER_TRIMMED", "OVER_TRIMMED"),
    "hojas_ext": ("OUTER_LEAVES_OK", "OUTER_DAMAGE_LIGHT", "OUTER_DAMAGE_MAJOR"),
    "color": ("COLOR_FRESH_OK", "YELLOW_BROWN_LIGHT", "YELLOW_BROWN_MAJOR", "WILTING"),
    "daño_fisico": ("PHYSICAL_OK", "TORN_LEAF", "BRUISE_CRUSH"),
    "daño_bio": ("BIO_OK", "ROT_DECAY", "VISIBLE_PEST", "PEST_DAMAGE", "DISEASE_VISIBLE"),
}

# Niveles posibles de la salida global (la columna admite además el vacío: sin etiquetar).
GLOBAL_LEVELS: tuple[str, ...] = ("ACCEPT", "REVIEW", "NON-CONFORMING")

# Vistas válidas de la columna ``vista`` (la columna admite además el vacío).
VISTAS: tuple[str, ...] = ("cenital", "lateral_a", "lateral_b", "base")

# Formas de conservación válidas de la columna ``conservacion`` (admite además el vacío).
CONSERVACIONES: tuple[str, ...] = ("nevera", "ambiente")

# Columnas esperadas de metadata.csv, en orden. Todas deben estar en la cabecera. Las ocho
# últimas son el contexto de captura: sus valores pueden ir vacíos.
METADATA_COLUMNS: tuple[str, ...] = (
    "image_id",
    "fuente",
    "licencia",
    "etiqueta_original",
    "lechuga_id",
    "vista",
    "sesion",
    *GROUPS,
    "salida_global",
    "variedad",  # texto libre
    "estado_seq",  # entero >= 0
    "horas_desde_corte",  # número >= 0
    "conservacion",  # una de CONSERVACIONES
    "temperatura_c",  # número (puede ser negativo)
    "intervencion",  # texto libre
    "dispositivo",  # texto libre
    "archivo_original",  # texto libre
)

# Columnas que nunca pueden quedar vacías (de dónde viene la imagen y con qué licencia).
_COLUMNAS_OBLIGATORIAS = ("fuente", "licencia")

# Columnas de lista cerrada que pueden ir vacías: columna -> valores permitidos.
_CERRADAS_OPCIONALES: dict[str, tuple[str, ...]] = {
    "vista": VISTAS,
    "conservacion": CONSERVACIONES,
}

# Número en notación decimal con punto (``12``, ``-2.5``, ``.5``, ``1e3``). Se comprueba con
# una expresión regular porque ``float()`` también deja pasar cosas como ``"1_0"`` o ``"nan"``.
_NUMERO_DECIMAL = re.compile(r"[+-]?([0-9]+(\.[0-9]*)?|\.[0-9]+)([eE][+-]?[0-9]+)?")


def is_blank(valor) -> bool:
    """Indica si una celda está vacía.

    En lenguaje llano: una celda cuenta como vacía si es ``None``, ``NaN`` (lo que pandas
    pone al leer una casilla en blanco de un CSV) o un texto que solo tiene espacios.
    """
    if valor is None:
        return True
    if isinstance(valor, str):
        return valor.strip() == ""
    try:
        return bool(pd.isna(valor))
    except (TypeError, ValueError):  # valores raros (listas, etc.): no son "vacío"
        return False


def _a_numero(valor) -> float | None:
    """Convierte una celda a número, o devuelve ``None`` si no es un número válido.

    En lenguaje llano: acepta números de Python o de numpy y texto con formato decimal con
    punto (``"12.5"``, ``" -2 "``). Rechaza los booleanos (``True`` no es un 1), la coma
    decimal (``"12,5"``), los textos que no son números y los valores no finitos (``nan``,
    ``inf``). Las celdas vacías no se tratan aquí: se comprueban antes con ``is_blank``.
    """
    if isinstance(valor, bool):
        return None
    if isinstance(valor, str):
        texto = valor.strip()
        if not _NUMERO_DECIMAL.fullmatch(texto):
            return None
        numero = float(texto)
    elif isinstance(valor, numbers.Real):
        numero = float(valor)
    else:
        return None
    return numero if math.isfinite(numero) else None


def _es_numero(valor) -> bool:
    """True si la celda es un número cualquiera (positivo, cero o negativo)."""
    return _a_numero(valor) is not None


def _es_numero_no_negativo(valor) -> bool:
    """True si la celda es un número mayor o igual que cero."""
    numero = _a_numero(valor)
    return numero is not None and numero >= 0


def _es_entero_no_negativo(valor) -> bool:
    """True si la celda es un número entero mayor o igual que cero.

    En lenguaje llano: ``3``, ``"3"``, ``3.0`` y ``"3.0"`` valen. Se acepta ``3.0`` porque
    ``pd.read_csv`` convierte en decimales una columna entera con huecos (``3`` pasa a
    ``3.0``), y eso no es un error del dato.
    """
    numero = _a_numero(valor)
    return numero is not None and numero >= 0 and numero.is_integer()


# Columnas numéricas que pueden ir vacías: columna -> (comprobación, requisito en texto).
# El texto usa ">=" y no "≥" a propósito: así se puede imprimir en una consola de Windows
# (cp1252) o redirigir a un archivo sin que falle la codificación.
_NUMERICAS_OPCIONALES = {
    "estado_seq": (_es_entero_no_negativo, "entero >= 0 o vacío"),
    "horas_desde_corte": (_es_numero_no_negativo, "número >= 0 con punto decimal, o vacío"),
    "temperatura_c": (_es_numero, "número con punto decimal, o vacío"),
}


def validate_metadata(df: pd.DataFrame) -> list[str]:
    """Comprueba que una tabla de metadatos cumple el esquema y lista los problemas.

    En lenguaje llano: revisa la tabla fila a fila y devuelve una lista con un texto por
    cada problema encontrado, indicando el ``image_id`` afectado (o ``fila N`` si la fila
    no tiene id). Si la lista está vacía, la tabla es válida. Nunca lanza excepciones por
    datos incorrectos ni modifica la tabla.

    Comprueba: columnas presentes (un único aviso que lista todas las que faltan);
    ``image_id`` presente y único; ``fuente`` y ``licencia`` no vacías; cada grupo con un
    valor de su lista o ``NO_VISIBLE`` (sin celdas vacías); ``salida_global`` vacía o uno de
    ``GLOBAL_LEVELS``; y el contexto de captura, cuyas celdas pueden ir vacías: ``vista``
    en ``VISTAS``, ``conservacion`` en ``CONSERVACIONES``, ``estado_seq`` entero >= 0,
    ``horas_desde_corte`` número >= 0 y ``temperatura_c`` número. Las columnas de texto libre
    (``variedad``, ``intervencion``, ``dispositivo``, ``archivo_original``) solo tienen que
    estar en la cabecera. Si faltan columnas, se siguen validando las que sí están.
    """
    errores: list[str] = []

    faltan = [columna for columna in METADATA_COLUMNS if columna not in df.columns]
    if len(faltan) == 1:
        errores.append(f"Falta 1 columna: {faltan[0]}")
    elif faltan:
        errores.append(f"Faltan {len(faltan)} columnas: {', '.join(faltan)}")

    indices = list(df.index)
    ids = df["image_id"].tolist() if "image_id" in df.columns else [None] * len(df)

    def quien(pos: int) -> str:
        """Texto que identifica la fila: su image_id o, si falta, su posición en la tabla."""
        return f"fila {indices[pos]}" if is_blank(ids[pos]) else f"image_id '{ids[pos]}'"

    # image_id: no vacío y sin repetir (los vacíos ya se avisan aparte y no cuentan).
    if "image_id" in df.columns:
        cuentas: dict = {}
        for pos, valor in enumerate(ids):
            if is_blank(valor):
                errores.append(f"{quien(pos)}: image_id vacío")
            else:
                cuentas[valor] = cuentas.get(valor, 0) + 1
        for valor, veces in cuentas.items():
            if veces > 1:
                errores.append(f"image_id '{valor}' duplicado ({veces} filas)")

    for columna in _COLUMNAS_OBLIGATORIAS:
        if columna in df.columns:
            for pos, valor in enumerate(df[columna].tolist()):
                if is_blank(valor):
                    errores.append(f"{quien(pos)}: '{columna}' vacío (es obligatorio)")

    for grupo, validos in GROUPS.items():
        if grupo not in df.columns:
            continue
        permitidos = (*validos, NO_VISIBLE)
        for pos, valor in enumerate(df[grupo].tolist()):
            if is_blank(valor):
                errores.append(f"{quien(pos)}: '{grupo}' vacío (usa {NO_VISIBLE} si no se ve)")
            elif valor not in permitidos:
                errores.append(
                    f"{quien(pos)}: valor '{valor}' no válido en '{grupo}' "
                    f"(válidos: {', '.join(permitidos)})"
                )

    if "salida_global" in df.columns:
        for pos, valor in enumerate(df["salida_global"].tolist()):
            if not is_blank(valor) and valor not in GLOBAL_LEVELS:
                errores.append(
                    f"{quien(pos)}: salida_global '{valor}' no válida "
                    f"(válidas: {', '.join(GLOBAL_LEVELS)} o vacío)"
                )

    # Contexto de captura: celdas vacías siempre válidas; si hay valor, debe cumplir la regla.
    for columna, validos in _CERRADAS_OPCIONALES.items():
        if columna not in df.columns:
            continue
        for pos, valor in enumerate(df[columna].tolist()):
            if not is_blank(valor) and valor not in validos:
                errores.append(
                    f"{quien(pos)}: valor '{valor}' no válido en '{columna}' "
                    f"(válidos: {', '.join(validos)} o vacío)"
                )

    for columna, (es_valido, requisito) in _NUMERICAS_OPCIONALES.items():
        if columna not in df.columns:
            continue
        for pos, valor in enumerate(df[columna].tolist()):
            if not is_blank(valor) and not es_valido(valor):
                errores.append(
                    f"{quien(pos)}: valor '{valor}' no válido en '{columna}' ({requisito})"
                )

    return errores
