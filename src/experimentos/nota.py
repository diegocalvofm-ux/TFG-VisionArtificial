"""Notas de experimento de Obsidian: crear desde la plantilla y editarlas sin tocar lo demás.

Una nota es un archivo Markdown con una cabecera YAML entre dos líneas ``---`` y secciones
``## Título``. Las funciones de aquí trabajan sobre el TEXTO de la nota y devuelven el texto
nuevo; ``modificar_nota`` se encarga de leer y escribir el archivo respetando sus saltos de
línea. La cabecera se lee con PyYAML pero se escribe cambiando solo las líneas afectadas,
porque reescribirla entera con PyYAML cambiaría el orden y el formato que Obsidian espera.
"""

import json
import re
from collections.abc import Callable, Collection, Mapping
from datetime import date
from decimal import Decimal
from pathlib import Path

import yaml

# La cabecera de estos campos no se edita por comando: el id lo da la numeración y las
# etiquetas son una lista (varias líneas).
CAMPOS_NO_EDITABLES = ("id", "tags")

# Tipos de la cabecera: "valor" es un número, "fecha" una fecha y todo lo demás, texto.
CAMPO_NUMERICO = "valor"
CAMPO_FECHA = "fecha"
_NUMERO = re.compile(r"^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$")

_NOMBRE_NOTA = re.compile(r"^exp(\d{3,})\.md$")
_MARCADOR_OBSIDIAN = re.compile(r"\{\{[^}]*\}\}")


def siguiente_id(carpeta: Path) -> str:
    """Devuelve el id de la siguiente nota (``exp001`` si la carpeta no tiene ninguna).

    En lenguaje llano: mira los archivos ``expNNN.md`` que hay, toma el número más alto y
    suma uno. Los archivos con otros nombres se ignoran.
    """
    numeros = [
        int(coincidencia.group(1))
        for ruta in Path(carpeta).glob("exp*.md")
        if (coincidencia := _NOMBRE_NOTA.match(ruta.name))
    ]
    return f"exp{(max(numeros) + 1 if numeros else 1):03d}"


def _cierre_de_cabecera(lineas: list[str]) -> int:
    """Posición de la línea ``---`` que cierra la cabecera. ``ValueError`` si no hay cabecera."""
    if not lineas or lineas[0] != "---":
        raise ValueError("la nota no tiene cabecera (debe empezar por una línea '---')")
    for posicion in range(1, len(lineas)):
        if lineas[posicion] == "---":
            return posicion
    raise ValueError("la cabecera de la nota no está cerrada (falta la segunda línea '---')")


def campos_de_plantilla(texto_plantilla: str) -> tuple[str, ...]:
    """Nombres de los campos de la cabecera de la plantilla, en su orden."""
    lineas = texto_plantilla.split("\n")
    cierre = _cierre_de_cabecera(lineas)
    claves = (re.match(r"^([A-Za-z_]\w*):", linea) for linea in lineas[1:cierre])
    return tuple(c.group(1) for c in claves if c)


def leer_cabecera(texto: str) -> dict:
    """Lee la cabecera de una nota como diccionario (los campos vacíos valen ``None``).

    Errores (ValueError): no hay cabecera, o no es YAML válido.
    """
    lineas = texto.split("\n")
    cierre = _cierre_de_cabecera(lineas)
    try:
        cabecera = yaml.safe_load("\n".join(lineas[1:cierre]))
    except yaml.YAMLError as error:
        raise ValueError(f"la cabecera no es YAML válido: {error}") from error
    if cabecera is None:
        return {}
    if not isinstance(cabecera, dict):
        raise ValueError("la cabecera debe ser una lista de 'campo: valor'")
    return cabecera


def _como_numero(clave: str, valor: str) -> str:
    """Escribe ``valor`` como un número que YAML lee como número (``ValueError`` si no lo es).

    ``1e-3`` se escribe ``0.001`` porque YAML lee ``1e-3`` como texto; ``0,35`` o ``85%``
    se rechazan en lugar de adivinar qué se quería decir.
    """
    if not _NUMERO.match(valor):
        raise ValueError(
            f"el campo '{clave}' debe ser un número con punto decimal, p. ej. 0.35 "
            f"(recibido: {valor!r})"
        )
    return format(Decimal(valor), "f")


def _como_fecha(clave: str, valor: str) -> str:
    """Valida una fecha ISO (AAAA-MM-DD) y la devuelve normalizada."""
    try:
        return date.fromisoformat(valor).isoformat()
    except ValueError:
        raise ValueError(
            f"el campo '{clave}' debe ser una fecha AAAA-MM-DD (recibido: {valor!r})"
        ) from None


def _formatear(clave: str, valor: str) -> str:
    """Escribe un valor de la cabecera como YAML válido y del tipo que corresponde al campo.

    En lenguaje llano: cada campo tiene un tipo. ``valor`` es un número y ``fecha`` una fecha;
    TODO lo demás es texto. Un texto que YAML leería como otra cosa (``1234567``, ``3.13``,
    ``true``, ``0123456``...) se escribe entre comillas para que al releer siga siendo
    exactamente el texto guardado: un commit o una versión no son números.
    """
    if "\n" in valor or "\r" in valor:
        raise ValueError(f"el valor de '{clave}' no puede tener saltos de línea")
    if valor == "":
        return ""
    if clave == CAMPO_NUMERICO:
        return _como_numero(clave, valor)
    if clave == CAMPO_FECHA:
        return _como_fecha(clave, valor)
    try:
        leido = yaml.safe_load(valor)
    except yaml.YAMLError:
        return json.dumps(valor, ensure_ascii=False)
    if isinstance(leido, str) and leido == valor:
        return valor
    return json.dumps(valor, ensure_ascii=False)  # el JSON entre comillas también es YAML


def poner_campos(texto: str, campos: Mapping[str, str], permitidos: Collection[str]) -> str:
    """Cambia el valor de ciertos campos de la cabecera y deja todo lo demás igual.

    En lenguaje llano: busca la línea ``campo: ...`` de cada clave y la reescribe. Si el
    campo no estaba en la nota (por ejemplo, una nota antigua), lo añade al final de la
    cabecera. No se toca ninguna otra línea ni el orden.

    ``permitidos`` son los campos que existen en la plantilla.

    Errores (ValueError): un campo que no está en ``permitidos`` (se listan los válidos), un
    campo no editable (``id``, ``tags``) o un valor con saltos de línea. Si algo falla no
    se cambia nada.
    """
    validos = sorted(c for c in permitidos if c not in CAMPOS_NO_EDITABLES)
    for clave in campos:
        if clave in CAMPOS_NO_EDITABLES:
            raise ValueError(f"el campo '{clave}' no se puede editar por comando")
        if clave not in validos:
            raise ValueError(
                f"el campo '{clave}' no existe en la plantilla (válidos: {', '.join(validos)})"
            )
    nuevas = {}
    for clave, valor in campos.items():
        formateado = _formatear(clave, valor)
        nuevas[clave] = f"{clave}: {formateado}" if formateado else f"{clave}:"

    lineas = texto.split("\n")
    cierre = _cierre_de_cabecera(lineas)
    for clave, linea_nueva in nuevas.items():
        patron = re.compile(rf"^{re.escape(clave)}:(\s|$)")
        posicion = next((i for i in range(1, cierre) if patron.match(lineas[i])), None)
        if posicion is None:
            lineas.insert(cierre, linea_nueva)
            cierre += 1
        else:
            lineas[posicion] = linea_nueva
    return "\n".join(lineas)


def _marcadores(nombre: str) -> tuple[str, str]:
    return f"<!-- registro:{nombre} -->", f"<!-- /registro:{nombre} -->"


def leer_bloque(texto: str, nombre: str) -> str | None:
    """Contenido del bloque automático ``nombre``, o ``None`` si la nota no lo tiene.

    Error (ValueError): el bloque tiene el marcador de inicio pero no el de fin.
    """
    inicio, fin = _marcadores(nombre)
    posicion = texto.find(inicio)
    if posicion == -1:
        return None
    final = texto.find(fin, posicion)
    if final == -1:
        raise ValueError(f"falta el marcador de cierre {fin}")
    return texto[posicion + len(inicio) : final].strip("\n")


def poner_bloque(texto: str, nombre: str, contenido: str, seccion: str) -> str:
    """Escribe un bloque automático en una sección de la nota, sin duplicarlo.

    En lenguaje llano: lo automático va entre dos marcadores (comentarios invisibles de
    Markdown). Si el bloque ya existe, se sustituye su contenido; si no, se inserta justo
    debajo del encabezado ``## seccion``. Lo que haya escrito el autor fuera de los
    marcadores no se toca, y repetir la operación con lo mismo no cambia nada.

    Errores (ValueError): la nota no tiene esa sección, o los marcadores están incompletos.
    """
    inicio, fin = _marcadores(nombre)
    # Una línea en blanco a cada lado de los marcadores: así ningún renderizador de Markdown
    # los pega a la tabla o al texto de al lado.
    bloque = f"{inicio}\n\n{contenido.strip(chr(10))}\n\n{fin}"

    posicion = texto.find(inicio)
    if posicion != -1:
        final = texto.find(fin, posicion)
        if final == -1:
            raise ValueError(f"falta el marcador de cierre {fin}")
        return texto[:posicion] + bloque + texto[final + len(fin) :]
    if fin in texto:
        raise ValueError(f"hay un marcador de cierre {fin} sin su marcador de inicio")

    encabezado = re.search(rf"^##[ \t]+{re.escape(seccion)}[ \t]*$", texto, re.MULTILINE)
    if encabezado is None:
        raise ValueError(f"la nota no tiene la sección '## {seccion}'")
    resto = texto[encabezado.end() :]
    resto = resto[1:] if resto.startswith("\n") else resto  # quitar el salto del encabezado
    separador = "" if resto == "" or resto.startswith("\n") else "\n"
    return texto[: encabezado.end()] + "\n\n" + bloque + "\n" + separador + resto


def leer_nota(ruta: Path) -> str:
    """Lee una nota en UTF-8, con los saltos de línea normalizados a ``\\n``."""
    with Path(ruta).open(encoding="utf-8", newline="") as archivo:
        return archivo.read().replace("\r\n", "\n")


def modificar_nota(ruta: Path, funcion: Callable[[str], str]) -> None:
    """Aplica ``funcion`` al texto de una nota y la guarda, conservando sus saltos de línea.

    En lenguaje llano: una nota creada en Windows puede usar saltos ``\\r\\n`` y otra
    ``\\n``; se respeta el que tuviera para que git no marque líneas cambiadas que no lo
    están. Si ``funcion`` falla, el archivo no se toca; si no cambia nada, tampoco se escribe.
    """
    ruta = Path(ruta)
    with ruta.open(encoding="utf-8", newline="") as archivo:
        crudo = archivo.read()
    saltos = "\r\n" if "\r\n" in crudo else "\n"
    texto = crudo.replace("\r\n", "\n")
    nuevo = funcion(texto)
    if nuevo != texto:
        with ruta.open("w", encoding="utf-8", newline="") as archivo:
            archivo.write(nuevo.replace("\n", saltos))


def crear_nota(
    carpeta: Path,
    plantilla: Path,
    campos: Mapping[str, str],
    hoy: date,
    despues: Callable[[str], str] | None = None,
) -> Path:
    """Crea la siguiente nota ``expNNN.md`` a partir de la plantilla y devuelve su ruta.

    En lenguaje llano: copia la plantilla, sustituye los marcadores de Obsidian
    (``{{title}}`` por el id y ``{{date:YYYY-MM-DD}}`` por la fecha de hoy) y rellena solo
    los campos que se den. Lo demás queda vacío: nunca se inventa un dato. ``despues``
    permite retocar el texto (p. ej. añadir un bloque) antes de escribirlo.

    Nunca sobrescribe una nota existente.

    Errores: ``ValueError`` (campo que no está en la plantilla o marcador desconocido) y
    ``FileExistsError`` (ya existe esa nota). Si algo falla no se crea ningún archivo.
    """
    texto = Path(plantilla).read_text(encoding="utf-8").replace("\r\n", "\n")
    editables = [c for c in campos_de_plantilla(texto) if c not in CAMPOS_NO_EDITABLES]
    id_nota = siguiente_id(carpeta)

    texto = texto.replace("{{title}}", id_nota).replace("{{date:YYYY-MM-DD}}", hoy.isoformat())
    desconocidos = _MARCADOR_OBSIDIAN.findall(texto)
    if desconocidos:
        raise ValueError(f"la plantilla tiene marcadores desconocidos: {', '.join(desconocidos)}")
    texto = poner_campos(texto, campos, editables)
    if despues is not None:
        texto = despues(texto)

    ruta = Path(carpeta) / f"{id_nota}.md"
    with ruta.open("x", encoding="utf-8", newline="\n") as archivo:  # "x": falla si ya existe
        archivo.write(texto)
    return ruta
