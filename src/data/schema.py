"""Esquema de los metadatos: grupos de calidad, valores válidos y validación.

Este módulo es la única fuente de verdad de los seis grupos y de sus valores. Otros
módulos (p. ej. la tabla de salida global) deben comprobarse contra lo definido aquí.
"""

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

# Columnas esperadas de metadata.csv, en orden.
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
)

# Columnas que nunca pueden quedar vacías (de dónde viene la imagen y con qué licencia).
_COLUMNAS_OBLIGATORIAS = ("fuente", "licencia")


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


def validate_metadata(df: pd.DataFrame) -> list[str]:
    """Comprueba que una tabla de metadatos cumple el esquema y lista los problemas.

    En lenguaje llano: revisa la tabla fila a fila y devuelve una lista con un texto por
    cada problema encontrado, indicando el ``image_id`` afectado (o ``fila N`` si la fila
    no tiene id). Si la lista está vacía, la tabla es válida. Nunca lanza excepciones por
    datos incorrectos ni modifica la tabla.

    Comprueba: columnas presentes; ``image_id`` presente y único; ``fuente`` y ``licencia``
    no vacías; cada grupo con un valor de su lista o ``NO_VISIBLE`` (sin celdas vacías);
    y ``salida_global`` vacía o uno de ``GLOBAL_LEVELS``. El resto de columnas no se valida.
    """
    errores: list[str] = []

    for columna in METADATA_COLUMNS:
        if columna not in df.columns:
            errores.append(f"Falta la columna '{columna}'")

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

    return errores
