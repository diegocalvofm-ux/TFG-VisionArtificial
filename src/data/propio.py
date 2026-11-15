"""Ingesta de las fotos propias: fotos + hoja de captura -> fotos limpias, metadata.csv e informe.

Qué hace, en lenguaje llano:

1. Lee las fotos de una carpeta. El nombre de cada una dice qué es: ``<lechuga>_<estado>_<vista>``
   (p. ej. ``LB-001_5_C.jpg``: batavia 1, estado 5, vista cenital).
2. Lee la hoja de captura: una fila por lechuga y estado con lo que se le hizo, los recuentos de
   hojas dañadas y amarillas, si el daño se ve en las fotos y notas.
3. Con las reglas del YAML (``configs/datasets/propio_piloto.yaml``) decide la etiqueta de cada
   uno de los seis grupos en cada foto, en este orden: por estado y variedad, por los recuentos
   de la hoja, según lo que se ve desde esa vista (y el "No" de la hoja) y, al final, las
   excepciones escritas a mano, que ganan sobre todo lo anterior.
4. Escribe las fotos giradas según su EXIF y sin metadatos, un ``informe.json`` y sus filas en
   ``metadata.csv``. Sustituye las filas de esa fuente y sesión y no toca las demás.

Una *toma* es el conjunto de fotos de una lechuga en un estado (hasta cuatro vistas).

Todo se comprueba antes de escribir nada. Los errores (nombres raros, fotos sin fila en la hoja,
textos que no se entienden...) se juntan en un único mensaje y entonces no se escribe ningún
archivo. Las reglas de etiquetado viven en el YAML: este código solo las aplica.
"""

import io
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.config import load_config
from src.data.hoja_captura import (
    leer_hoja,
    leer_recuento,
    leer_se_ve,
    normalizar,
    texto_celda,
    traducir_estado,
)
from src.data.publicos import comprobar_metadata, escribir_metadata, leer_metadata
from src.data.schema import (
    CONSERVACIONES,
    GLOBAL_LEVELS,
    GROUPS,
    METADATA_COLUMNS,
    NO_VISIBLE,
    VISTAS,
    is_blank,
)
from src.evaluation.aggregate import aggregate
from src.pipeline.imagenes import guardar_imagen, leer_imagen

# En por_estado, este valor significa "la etiqueta del estado 0 de esa variedad".
HEREDA_ESTADO_0 = "estado_0"
EXTENSIONES_FOTO = (".jpg", ".jpeg")
INFORME = "informe.json"

_NOMBRE = re.compile(r"[A-Za-z0-9_-]+")
_CODIGO = re.compile(r"[A-Za-z0-9]+")  # sin "_": es el separador del nombre de archivo
_PREFIJO = re.compile(r"[A-Z]+")
_CLAVES = frozenset(
    ("fuente", "sesion", "licencia", "dispositivo", "fotos", "hoja", "hoja_nombre",
     "excepciones", "salida", "columnas", "variedades", "vistas", "conservacion", "estados",
     "serie", "estado_0", "por_estado", "recuentos", "visibilidad", "no_se_ve")
)  # fmt: skip
_OPCIONALES = frozenset(("dispositivo", "excepciones"))
# Columnas de la hoja que la ingesta necesita siempre (las de los recuentos las nombra el YAML).
_COLUMNAS_HOJA = ("lechuga", "variedad", "estado", "se_ve_dano", "notas")
_COLUMNAS_EXCEPCIONES = ("lechuga", "estado", "grupo", "etiqueta")


@dataclass(frozen=True)
class Reglas:
    """Reglas del YAML ya validadas y listas para usar (ver ``validar_reglas``)."""

    fuente: str
    sesion: str
    licencia: str
    dispositivo: str
    columnas: dict[str, str]  # clave interna -> cabecera de la hoja
    variedades: dict[str, str]  # prefijo del lechuga_id -> variedad
    vistas: dict[str, str]  # letra del nombre -> vista del esquema
    conservacion: dict[str, str]  # lechuga_id -> nevera / ambiente
    estados: tuple[str, ...]  # códigos en el orden del YAML
    alias: dict[str, str]  # texto normalizado -> código
    serie: dict[str, float]  # código de la serie temporal -> horas desde el corte
    estado_0: dict[str, dict[str, str]]  # variedad -> grupo -> etiqueta
    por_estado: dict[str, dict[str, str]]  # grupo -> código -> etiqueta (o HEREDA_ESTADO_0)
    recuentos: dict[str, tuple[str, tuple[str, str, str]]]  # grupo -> (columna, 3 etiquetas)
    leve_desde: int
    grave_desde: int
    palabras: dict[str, int]  # palabra normalizada -> recuento ("toda" -> 5)
    visibilidad: dict[str, frozenset[str]]  # grupo -> letras de las vistas en que se ve
    no_se_ve_grupo: str
    no_se_ve_sin_defecto: str
    patron_lechuga: re.Pattern
    patron_nombre: re.Pattern  # nombre de archivo sin extensión, con grupos lechuga/estado/letra


@dataclass(frozen=True)
class Config:
    """Reglas más las rutas de una ingesta (ya resueltas contra la raíz del repositorio)."""

    reglas: Reglas
    fotos: Path
    hoja: Path
    hoja_nombre: str
    excepciones: Path | None
    salida: Path


@dataclass(frozen=True)
class Foto:
    """Una foto de la carpeta, con lo que dice su nombre."""

    nombre: str  # nombre del archivo, con extensión
    image_id: str  # nombre sin extensión
    lechuga: str
    estado: str
    letra: str  # letra de la vista (C, F, G, B)

    @property
    def toma(self) -> tuple[str, str]:
        return (self.lechuga, self.estado)


@dataclass(frozen=True)
class FilaHoja:
    """Una fila válida de la hoja de captura, ya traducida."""

    fila: int  # número de fila en Excel
    lechuga: str
    variedad: str  # la del prefijo del lechuga_id, según el YAML
    estado: str  # código
    estado_texto: str  # el texto del estado tal como viene en la hoja
    recuentos: dict[str, int | None]  # columna de recuento -> número (None = desconocido)
    se_ve: bool | None  # respuesta a "¿Se ve el daño en las fotos?"
    notas: str


@dataclass(frozen=True)
class Excepcion:
    """Una línea del CSV de excepciones: fija la etiqueta de un grupo en una toma."""

    linea: int
    lechuga: str
    estado: str
    grupo: str
    etiqueta: str
    letra: str | None  # None = todas las vistas


@dataclass
class Cruce:
    """Resultado de cruzar las fotos con las filas de la hoja."""

    tomas: dict[tuple[str, str], list[Foto]]  # en orden de lechuga y estado
    errores: list[str]
    filas_sin_fotos: list[dict]
    incompletas: list[dict]


# --- Reglas del YAML ----------------------------------------------------------------------


def _error(texto: str) -> ValueError:
    return ValueError(f"Config de la ingesta: {texto}")


def _diccionario(config: dict, clave: str, vacio: bool = False) -> dict:
    """``config[clave]`` como diccionario con claves de texto; error si no lo es."""
    valor = config.get(clave)
    if not isinstance(valor, dict) or (not valor and not vacio):
        raise _error(f"'{clave}' debe ser un diccionario (clave: valor) no vacío")
    return {str(k): v for k, v in valor.items()}


def _comprobar_etiqueta(grupo: str, etiqueta, donde: str) -> None:
    """Error si el grupo no existe o la etiqueta no es de ese grupo (``NO_VISIBLE`` vale)."""
    if grupo not in GROUPS:
        raise _error(f"{donde}: grupo '{grupo}' no existe (válidos: {', '.join(GROUPS)})")
    if etiqueta != NO_VISIBLE and etiqueta not in GROUPS[grupo]:
        raise _error(
            f"{donde}: etiqueta '{etiqueta}' no válida en '{grupo}' "
            f"(válidas: {', '.join((*GROUPS[grupo], NO_VISIBLE))})"
        )


def _alternativas(valores) -> str:
    """Alternativas de una regex (``a|b|c``), las largas primero para que no las tape una corta."""
    return "|".join(re.escape(v) for v in sorted(valores, key=len, reverse=True))


def _validar_claves(config) -> None:
    """Error si faltan claves obligatorias o hay alguna desconocida (una errata no pasa)."""
    if not isinstance(config, dict):
        raise _error("debe ser un diccionario YAML")
    desconocidas = sorted(set(map(str, config)) - _CLAVES)
    if desconocidas:
        raise _error(f"claves desconocidas: {', '.join(desconocidas)}")
    faltan = sorted(c for c in _CLAVES - _OPCIONALES if c not in config)
    if faltan:
        raise _error(f"faltan las claves: {', '.join(faltan)}")
    for clave in ("fuente", "sesion"):
        if not isinstance(config[clave], str) or not _NOMBRE.fullmatch(config[clave]):
            raise _error(f"'{clave}' debe ser un nombre sin espacios (letras, números, - y _)")
    if not isinstance(config["licencia"], str) or not config["licencia"].strip():
        raise _error("'licencia' no puede ir vacía")


def _validar_columnas(config: dict) -> dict[str, str]:
    """``columnas``: clave interna -> cabecera de la hoja, con las claves que la ingesta usa."""
    columnas = _diccionario(config, "columnas")
    if any(not isinstance(v, str) or not v.strip() for v in columnas.values()):
        raise _error("columnas: cada clave necesita el texto de su cabecera en la hoja")
    faltan = [c for c in _COLUMNAS_HOJA if c not in columnas]
    if faltan:
        raise _error(f"columnas: faltan las claves {', '.join(faltan)}")
    return {clave: cabecera.strip() for clave, cabecera in columnas.items()}


def _validar_identidad(config: dict) -> tuple[dict, re.Pattern, dict, dict]:
    """``variedades``, ``vistas`` y ``conservacion``. Devuelve también la regex del lechuga_id.

    En lenguaje llano: los prefijos (LI, LB) van en mayúsculas, cada letra de vista apunta a una
    vista distinta del esquema y la conservación de cada lechuga es una de las del esquema.
    """
    variedades = {k: str(v) for k, v in _diccionario(config, "variedades").items()}
    for prefijo in variedades:
        if not _PREFIJO.fullmatch(prefijo):
            raise _error(f"variedades: el prefijo '{prefijo}' debe ir en mayúsculas (p. ej. LI)")
    patron_lechuga = re.compile(rf"(?:{_alternativas(variedades)})-[0-9]{{3}}")

    vistas = _diccionario(config, "vistas")
    for letra, vista in vistas.items():
        if not _CODIGO.fullmatch(letra):
            raise _error(f"vistas: la letra '{letra}' no es válida")
        if vista not in VISTAS:
            raise _error(f"vistas: '{vista}' no es una vista del esquema ({', '.join(VISTAS)})")
        repetidas = [otra for otra, v in vistas.items() if v == vista]
        if len(repetidas) > 1:
            raise _error(f"vistas: la vista '{vista}' está repetida ({', '.join(repetidas)})")

    conservacion = _diccionario(config, "conservacion", vacio=True)
    for lechuga, valor in conservacion.items():
        if not patron_lechuga.fullmatch(lechuga):
            raise _error(f"conservacion: '{lechuga}' no es un lechuga_id válido")
        if valor not in CONSERVACIONES:
            raise _error(
                f"conservacion: '{valor}' no válida para {lechuga} "
                f"(válidas: {', '.join(CONSERVACIONES)})"
            )
    return variedades, patron_lechuga, vistas, conservacion


def _validar_estados(config: dict) -> tuple[tuple[str, ...], dict[str, float], dict[str, str]]:
    """``estados`` y ``serie``. Devuelve (códigos, horas de la serie, alias normalizados).

    En lenguaje llano: cada código tiene letras y números (sin ``_``, que separa las partes del
    nombre de archivo) y es un número o está en ``serie``, porque ``estado_seq`` es un entero.
    Cada estado lista los textos con que puede venir en la hoja; se guardan normalizados junto
    con el propio código, y ninguno puede servir para dos estados.
    """
    crudos = _diccionario(config, "estados")
    estados = tuple(crudos)
    serie = {}
    for codigo, horas in _diccionario(config, "serie", vacio=True).items():
        if codigo not in estados:
            raise _error(f"serie: el estado '{codigo}' no está en 'estados'")
        if isinstance(horas, bool) or not isinstance(horas, int | float) or horas < 0:
            raise _error(f"serie: las horas de '{codigo}' deben ser un número >= 0")
        serie[codigo] = horas

    alias: dict[str, str] = {}
    for codigo, textos in crudos.items():
        if not _CODIGO.fullmatch(codigo):
            raise _error(f"estados: el código '{codigo}' solo puede tener letras y números")
        if not codigo.isdigit() and codigo not in serie:
            raise _error(
                f"estados: el código '{codigo}' no es un número ni está en 'serie' "
                "(estado_seq tiene que ser un entero)"
            )
        if not isinstance(textos, list) or not textos:
            raise _error(f"estados: '{codigo}' necesita una lista de textos")
        for texto in (codigo, *textos):
            clave = normalizar(texto)
            if not clave:
                raise _error(f"estados: '{codigo}' tiene un texto vacío")
            previo = alias.setdefault(clave, codigo)
            if previo != codigo:
                raise _error(f"estados: el texto '{clave}' está en los estados {previo} y {codigo}")
    return estados, serie, alias


def _validar_por_estado(
    config: dict, estados: tuple[str, ...], variedades: dict[str, str]
) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    """``estado_0`` y ``por_estado``. Devuelve (estado_0, por_estado).

    En lenguaje llano: ``estado_0`` da, para cada variedad, las etiquetas del estado "tal como
    viene". ``por_estado`` da, para cada grupo, una etiqueta por estado (todos los estados, ni
    uno más); el valor ``estado_0`` remite a la de su variedad y solo vale en los grupos que
    ``estado_0`` define para todas las variedades.
    """
    crudo = _diccionario(config, "estado_0")
    nombres = sorted(variedades.values())
    if set(crudo) != set(nombres):
        raise _error(f"estado_0: necesita una entrada por variedad ({', '.join(nombres)})")
    estado_0: dict[str, dict[str, str]] = {}
    for variedad, grupos in crudo.items():
        if not isinstance(grupos, dict):
            raise _error(f"estado_0.{variedad}: debe ser grupo: etiqueta")
        for grupo, etiqueta in grupos.items():
            _comprobar_etiqueta(grupo, etiqueta, f"estado_0.{variedad}")
        estado_0[variedad] = dict(grupos)
    con_estado_0 = sorted(set.intersection(*(set(g) for g in estado_0.values())))

    por_estado: dict[str, dict[str, str]] = {}
    for grupo, por_codigo in _diccionario(config, "por_estado").items():
        if grupo not in GROUPS:
            raise _error(f"por_estado: grupo '{grupo}' no existe (válidos: {', '.join(GROUPS)})")
        if not isinstance(por_codigo, dict):
            raise _error(f"por_estado.{grupo}: debe ser estado: etiqueta")
        por_codigo = {str(k): v for k, v in por_codigo.items()}
        faltan = [f"'{c}'" for c in estados if c not in por_codigo]
        if faltan:
            raise _error(f"por_estado.{grupo}: faltan estados: {', '.join(faltan)}")
        sobran = [f"'{c}'" for c in por_codigo if c not in estados]
        if sobran:
            raise _error(
                f"por_estado.{grupo}: estados que no están en 'estados': {', '.join(sobran)}"
            )
        for codigo, etiqueta in por_codigo.items():
            if etiqueta != HEREDA_ESTADO_0:
                _comprobar_etiqueta(grupo, etiqueta, f"por_estado.{grupo}.{codigo}")
            elif grupo not in con_estado_0:
                raise _error(
                    f"por_estado.{grupo}: '{HEREDA_ESTADO_0}' solo vale en los grupos que define "
                    f"estado_0 para todas las variedades ({', '.join(con_estado_0)})"
                )
        por_estado[grupo] = por_codigo
    return estado_0, por_estado


def _validar_recuentos(config: dict, columnas: dict[str, str]) -> dict:
    """``recuentos``: umbrales, palabras y grupos que salen de un recuento de la hoja.

    En lenguaje llano: los umbrales cumplen ``0 < leve_desde < grave_desde``, cada palabra vale
    un entero >= 0 y cada grupo nombra una columna de ``columnas`` y tres etiquetas de su grupo
    (sin daño, leve y grave). Devuelve un diccionario con ``leve_desde``, ``grave_desde``,
    ``palabras`` (normalizadas) y ``grupos`` (grupo -> (columna, etiquetas)).
    """
    crudo = _diccionario(config, "recuentos")
    sobran = sorted(set(crudo) - {"leve_desde", "grave_desde", "palabras", "grupos"})
    if sobran:
        raise _error(f"recuentos: claves desconocidas: {', '.join(sobran)}")
    leve, grave = crudo.get("leve_desde"), crudo.get("grave_desde")
    enteros = all(isinstance(n, int) and not isinstance(n, bool) for n in (leve, grave))
    if not enteros or not 0 < leve < grave:
        raise _error(
            "recuentos: leve_desde y grave_desde deben ser enteros con "
            f"0 < leve_desde < grave_desde (llegó {leve!r} y {grave!r})"
        )
    palabras = {}
    for palabra, numero in (crudo.get("palabras") or {}).items():
        if isinstance(numero, bool) or not isinstance(numero, int) or numero < 0:
            raise _error(f"recuentos.palabras: '{palabra}' debe valer un entero >= 0")
        palabras[normalizar(palabra)] = numero

    grupos_crudos = crudo.get("grupos")
    if not isinstance(grupos_crudos, dict) or not grupos_crudos:
        raise _error("recuentos: 'grupos' debe ser grupo: {columna, etiquetas}")
    grupos = {}
    for grupo, datos in grupos_crudos.items():
        if grupo not in GROUPS:
            raise _error(f"recuentos.grupos: grupo '{grupo}' no existe")
        columna = datos.get("columna") if isinstance(datos, dict) else None
        if columna not in columnas:
            raise _error(f"recuentos.grupos.{grupo}: la columna '{columna}' no está en 'columnas'")
        etiquetas = datos.get("etiquetas")
        if not isinstance(etiquetas, list) or len(etiquetas) != 3:
            raise _error(f"recuentos.grupos.{grupo}: necesita 3 etiquetas (sin daño, leve y grave)")
        for etiqueta in etiquetas:
            _comprobar_etiqueta(grupo, etiqueta, f"recuentos.grupos.{grupo}")
        grupos[grupo] = (columna, tuple(etiquetas))
    return {"leve_desde": leve, "grave_desde": grave, "palabras": palabras, "grupos": grupos}


def _validar_visibilidad(config: dict, vistas: dict[str, str]) -> dict[str, frozenset[str]]:
    """``visibilidad``: los seis grupos, cada uno con las letras de ``vistas`` en que se ve."""
    crudo = _diccionario(config, "visibilidad")
    sobran = [g for g in crudo if g not in GROUPS]
    if sobran:
        raise _error(f"visibilidad: grupos que no existen: {', '.join(sobran)}")
    faltan = [g for g in GROUPS if g not in crudo]
    if faltan:
        raise _error(f"visibilidad: faltan grupos: {', '.join(faltan)}")
    visibilidad = {}
    for grupo in GROUPS:
        letras = crudo[grupo]
        if not isinstance(letras, list):
            raise _error(f"visibilidad.{grupo}: debe ser una lista de vistas")
        for letra in letras:
            if letra not in vistas:
                raise _error(
                    f"visibilidad.{grupo}: la vista '{letra}' no está en 'vistas' "
                    f"({', '.join(vistas)})"
                )
        visibilidad[grupo] = frozenset(letras)
    return visibilidad


def validar_reglas(config: dict) -> Reglas:
    """Comprueba las reglas de etiquetado del YAML (sin mirar el disco) y las devuelve listas.

    En lenguaje llano: antes de etiquetar nada se revisa que las reglas tienen sentido y se
    para con un mensaje claro si algo no cuadra. Cada sección la revisa su función
    (``_validar_*``); aquí, además, se comprueba que los seis grupos de ``GROUPS`` salen o de
    ``por_estado`` o de ``recuentos`` (nunca de los dos ni de ninguno) y que la regla del "No"
    (``no_se_ve``) nombra un grupo y su etiqueta sin defecto.

    Devuelve un ``Reglas`` con los textos ya normalizados y las expresiones regulares del
    ``lechuga_id`` y del nombre de archivo. Lanza ``ValueError``.
    """
    _validar_claves(config)
    columnas = _validar_columnas(config)
    variedades, patron_lechuga, vistas, conservacion = _validar_identidad(config)
    estados, serie, alias = _validar_estados(config)
    estado_0, por_estado = _validar_por_estado(config, estados, variedades)
    recuentos = _validar_recuentos(config, columnas)

    ambos = [g for g in GROUPS if g in por_estado and g in recuentos["grupos"]]
    if ambos:
        raise _error(f"los grupos {', '.join(ambos)} están a la vez en por_estado y en recuentos")
    faltan = [g for g in GROUPS if g not in por_estado and g not in recuentos["grupos"]]
    if faltan:
        raise _error(f"faltan grupos en por_estado o en recuentos: {', '.join(faltan)}")
    visibilidad = _validar_visibilidad(config, vistas)

    crudo = _diccionario(config, "no_se_ve")
    grupo, sin_defecto = crudo.get("grupo"), crudo.get("sin_defecto")
    if grupo not in GROUPS:
        raise _error(f"no_se_ve: grupo '{grupo}' no existe")
    if sin_defecto not in GROUPS[grupo]:
        raise _error(f"no_se_ve: '{sin_defecto}' no es una etiqueta de '{grupo}'")

    patron_nombre = re.compile(
        rf"(?P<lechuga>{patron_lechuga.pattern})_(?P<estado>{_alternativas(estados)})"
        rf"_(?P<letra>{_alternativas(vistas)})"
    )
    dispositivo = config.get("dispositivo")
    return Reglas(
        fuente=config["fuente"],
        sesion=config["sesion"],
        licencia=config["licencia"].strip(),
        dispositivo="" if dispositivo is None else str(dispositivo).strip(),
        columnas=columnas,
        variedades=variedades,
        vistas=vistas,
        conservacion=conservacion,
        estados=estados,
        alias=alias,
        serie=serie,
        estado_0=estado_0,
        por_estado=por_estado,
        recuentos=recuentos["grupos"],
        leve_desde=recuentos["leve_desde"],
        grave_desde=recuentos["grave_desde"],
        palabras=recuentos["palabras"],
        visibilidad=visibilidad,
        no_se_ve_grupo=grupo,
        no_se_ve_sin_defecto=sin_defecto,
        patron_lechuga=patron_lechuga,
        patron_nombre=patron_nombre,
    )


def cargar_config(ruta: str | Path, raiz: str | Path) -> Config:
    """Lee el YAML de la ingesta, valida sus reglas y resuelve sus rutas contra ``raiz``.

    Comprueba también que existen la carpeta de fotos, la hoja y (si se da) el CSV de
    excepciones; si no, lanza ``FileNotFoundError`` con la ruta relativa del YAML.
    """
    raiz = Path(raiz)
    config = load_config(ruta)
    reglas = validar_reglas(config)

    def ruta_de(clave: str) -> Path:
        valor = config.get(clave)
        if not isinstance(valor, str) or not valor.strip():
            raise _error(f"'{clave}' debe ser una ruta relativa a la raíz del repositorio")
        return raiz / valor

    fotos, hoja, salida = ruta_de("fotos"), ruta_de("hoja"), ruta_de("salida")
    if not fotos.is_dir():
        raise FileNotFoundError(f"No se encuentra la carpeta de fotos {config['fotos']}")
    if not hoja.is_file():
        raise FileNotFoundError(f"No se encuentra la hoja de captura {config['hoja']}")
    if not isinstance(config["hoja_nombre"], str) or not config["hoja_nombre"].strip():
        raise _error("'hoja_nombre' debe ser el nombre de la hoja del xlsx")
    excepciones = None
    if not is_blank(config.get("excepciones")):
        excepciones = ruta_de("excepciones")
        if not excepciones.is_file():
            raise FileNotFoundError(
                f"No se encuentra el CSV de excepciones {config['excepciones']}"
            )
    return Config(
        reglas=reglas,
        fotos=fotos,
        hoja=hoja,
        hoja_nombre=config["hoja_nombre"],
        excepciones=excepciones,
        salida=salida,
    )


# --- Fotos y hoja -------------------------------------------------------------------------


def leer_fotos(carpeta: str | Path, reglas: Reglas) -> tuple[list[Foto], list[str]]:
    """Lee los nombres de la carpeta de fotos. Devuelve (fotos ordenadas por nombre, errores).

    En lenguaje llano: cada archivo tiene que llamarse ``<lechuga>_<estado>_<vista>.jpg`` (o
    ``.jpeg``, en mayúsculas o minúsculas) con los prefijos, estados y letras del YAML. Todo lo
    que no encaje (otro nombre, otra extensión, una subcarpeta) se junta en UN error que lo
    nombra todo. También es error que dos archivos den el mismo ``image_id``
    (``X.jpg`` y ``X.jpeg``); esas fotos no se devuelven.
    """
    carpeta = Path(carpeta)
    fotos, malos = [], []
    for ruta in sorted(carpeta.iterdir(), key=lambda p: p.name):
        coincidencia = reglas.patron_nombre.fullmatch(ruta.stem)
        if not ruta.is_file() or ruta.suffix.lower() not in EXTENSIONES_FOTO or not coincidencia:
            malos.append(ruta.name + ("/" if ruta.is_dir() else ""))
            continue
        fotos.append(
            Foto(
                nombre=ruta.name,
                image_id=ruta.stem,
                lechuga=coincidencia["lechuga"],
                estado=coincidencia["estado"],
                letra=coincidencia["letra"],
            )
        )

    errores = []
    if malos:
        errores.append(
            f"{len(malos)} archivo(s) de la carpeta de fotos no siguen el formato "
            f"<lechuga>_<estado>_<vista>.jpg (lechuga {'/'.join(reglas.variedades)}-NNN, estado "
            f"{', '.join(reglas.estados)}, vista {', '.join(reglas.vistas)}): {', '.join(malos)}"
        )
    por_id = defaultdict(list)
    for foto in fotos:
        por_id[foto.image_id].append(foto.nombre)
    repetidos = {i for i, nombres in por_id.items() if len(nombres) > 1}
    for image_id in sorted(repetidos):
        errores.append(f"image_id '{image_id}' repetido: {', '.join(por_id[image_id])}")
    fotos = [f for f in fotos if f.image_id not in repetidos]
    if not fotos and not errores:
        errores.append("la carpeta de fotos no tiene ninguna foto")
    return fotos, errores


def filas_de_hoja(
    crudas: list[dict], reglas: Reglas
) -> tuple[list[FilaHoja], list[dict], list[str]]:
    """Traduce las filas leídas de la hoja. Devuelve (filas válidas, filas ignoradas, errores).

    En lenguaje llano, por cada fila:

    - si alguna celda tiene solo guiones, ese estado no se hizo: se ignora (y se anota);
    - la lechuga tiene que tener el formato del YAML (``LI-001``; da igual mayúsculas);
    - el estado se traduce a su código con los textos de ``estados`` (si no se entiende, error);
    - la variedad, si viene, tiene que coincidir con la del prefijo de la lechuga;
    - los recuentos se leen con ``leer_recuento`` (vacío = desconocido) y "¿Se ve el daño?" con
      ``leer_se_ve``;
    - una lechuga no puede tener dos filas del mismo estado.

    Los errores nombran la fila de Excel y la columna para poder corregirlos en la hoja.
    """
    filas, ignoradas, errores = [], [], []
    vistas: dict[tuple[str, str], int] = {}
    columnas_recuento = sorted({columna for columna, _ in reglas.recuentos.values()})
    for cruda in crudas:
        numero = cruda["fila"]
        lechuga = texto_celda(cruda["lechuga"]).upper()
        codigo = traducir_estado(cruda["estado"], reglas.alias)
        if cruda["no_hecha"]:
            estado = codigo or texto_celda(cruda["estado"])
            ignoradas.append({"fila": numero, "lechuga": lechuga, "estado": estado})
            continue

        problemas = []
        variedad = None
        if not lechuga:
            problemas.append(f"'{reglas.columnas['lechuga']}' vacía")
        elif not reglas.patron_lechuga.fullmatch(lechuga):
            problemas.append(f"lechuga '{lechuga}' no válida (formato como LI-001 o LB-001)")
        else:
            prefijo = lechuga.split("-")[0]
            variedad = reglas.variedades[prefijo]
            escrita = texto_celda(cruda["variedad"])
            if escrita and normalizar(escrita) != normalizar(variedad):
                problemas.append(
                    f"la variedad '{escrita}' no coincide con la del prefijo {prefijo} ({variedad})"
                )
        estado_texto = texto_celda(cruda["estado"])
        if not estado_texto:
            problemas.append(f"'{reglas.columnas['estado']}' vacío")
        elif codigo is None:
            problemas.append(
                f"estado '{estado_texto}' no reconocido (si es otra forma de escribir un estado, "
                "añádela a 'estados' del YAML)"
            )
        recuentos = {}
        for columna in columnas_recuento:
            try:
                recuentos[columna] = leer_recuento(cruda[columna], reglas.palabras)
            except ValueError as error:
                problemas.append(f"'{reglas.columnas[columna]}': {error}")
        se_ve = None
        try:
            se_ve = leer_se_ve(cruda["se_ve_dano"])
        except ValueError as error:
            problemas.append(f"'{reglas.columnas['se_ve_dano']}': {error}")
        if problemas:
            errores += [f"hoja, fila {numero}: {problema}" for problema in problemas]
            continue

        if (lechuga, codigo) in vistas:
            errores.append(
                f"hoja, filas {vistas[(lechuga, codigo)]} y {numero}: {lechuga}, estado {codigo} "
                "repetido"
            )
            continue
        vistas[(lechuga, codigo)] = numero
        filas.append(
            FilaHoja(
                fila=numero,
                lechuga=lechuga,
                variedad=variedad,
                estado=codigo,
                estado_texto=estado_texto,
                recuentos=recuentos,
                se_ve=se_ve,
                notas=texto_celda(cruda["notas"]),
            )
        )
    return filas, ignoradas, errores


def _orden_toma(reglas: Reglas):
    """Clave para ordenar tomas: por lechuga y luego por el orden de los estados en el YAML."""
    return lambda toma: (toma[0], reglas.estados.index(toma[1]))


def cruzar(
    fotos: list[Foto], filas: list[FilaHoja], ignoradas: list[dict], reglas: Reglas
) -> Cruce:
    """Junta las fotos de cada toma y las compara con las filas de la hoja.

    En lenguaje llano:

    - una toma con fotos y sin fila en la hoja es un ERROR (no se sabría cómo etiquetarla); si
      su fila existe pero tiene guiones, el mensaje lo dice;
    - una fila sin fotos es un AVISO (quizá ese estado aún no se ha fotografiado);
    - una toma con menos vistas que las del YAML es un AVISO con las vistas que faltan.
    """
    vistas = list(reglas.vistas)
    tomas: dict[tuple[str, str], list[Foto]] = defaultdict(list)
    for foto in fotos:
        tomas[foto.toma].append(foto)
    tomas = {
        toma: sorted(tomas[toma], key=lambda f: vistas.index(f.letra))
        for toma in sorted(tomas, key=_orden_toma(reglas))
    }
    con_fila = {(f.lechuga, f.estado) for f in filas}
    con_guiones = {(i["lechuga"], i["estado"]): i["fila"] for i in ignoradas}

    errores = []
    for toma, fotos_toma in tomas.items():
        if toma in con_fila:
            continue
        nombres = ", ".join(sorted(f.nombre for f in fotos_toma))
        pista = ""
        if toma in con_guiones:
            pista = f"; la fila {con_guiones[toma]} tiene guiones y se ignora"
        errores.append(f"{toma[0]}, estado {toma[1]}: fotos sin fila en la hoja ({nombres}){pista}")

    filas_sin_fotos = [
        {"fila": f.fila, "lechuga": f.lechuga, "estado": f.estado}
        for f in filas
        if (f.lechuga, f.estado) not in tomas
    ]
    incompletas = []
    for (lechuga, estado), fotos_toma in tomas.items():
        presentes = {f.letra for f in fotos_toma}
        faltan = [letra for letra in vistas if letra not in presentes]
        if faltan:
            incompletas.append({"lechuga": lechuga, "estado": estado, "faltan": faltan})
    return Cruce(tomas, errores, filas_sin_fotos, incompletas)


# --- Etiquetado ---------------------------------------------------------------------------


def etiquetas_de_estado(fila: FilaHoja, reglas: Reglas) -> tuple[dict[str, str], list[str]]:
    """Etiquetas de los seis grupos de una toma, antes de mirar la vista. Devuelve
    (etiquetas, grupos con recuento desconocido).

    En lenguaje llano: los grupos de ``por_estado`` toman la etiqueta que el YAML da a ese
    estado (``estado_0`` = la del estado 0 de su variedad: así 24h y 48h heredan corte y recorte).
    Los grupos de ``recuentos`` miran el número de la hoja: menos de ``leve_desde`` es la
    primera etiqueta (sin daño), desde ``leve_desde`` la segunda (leve) y desde ``grave_desde``
    la tercera (grave). Un recuento desconocido deja el grupo en ``NO_VISIBLE``.
    """
    etiquetas = {}
    for grupo, por_codigo in reglas.por_estado.items():
        etiqueta = por_codigo[fila.estado]
        if etiqueta == HEREDA_ESTADO_0:
            etiqueta = reglas.estado_0[fila.variedad][grupo]
        etiquetas[grupo] = etiqueta
    desconocidos = []
    for grupo, (columna, (sin_dano, leve, grave)) in reglas.recuentos.items():
        numero = fila.recuentos[columna]
        if numero is None:
            etiquetas[grupo] = NO_VISIBLE
            desconocidos.append(grupo)
        elif numero >= reglas.grave_desde:
            etiquetas[grupo] = grave
        elif numero >= reglas.leve_desde:
            etiquetas[grupo] = leve
        else:
            etiquetas[grupo] = sin_dano
    return {g: etiquetas[g] for g in GROUPS}, [g for g in GROUPS if g in desconocidos]


def etiquetas_de_foto(base: dict[str, str], letra: str, se_ve: bool | None, reglas: Reglas) -> dict:
    """Etiquetas de una foto concreta a partir de las de su toma.

    En lenguaje llano: un grupo que no se ve desde esa vista (matriz ``visibilidad`` del YAML)
    pasa a ``NO_VISIBLE``. Además, si la hoja dice "No" en "¿Se ve el daño en las fotos?" y el
    grupo de ``no_se_ve`` tiene un defecto (una etiqueta distinta de ``sin_defecto``), ese grupo
    pasa a ``NO_VISIBLE`` en todas las vistas. Si no hay defecto, el "No" no cambia nada.
    """
    etiquetas = {
        grupo: etiqueta if letra in reglas.visibilidad[grupo] else NO_VISIBLE
        for grupo, etiqueta in base.items()
    }
    grupo = reglas.no_se_ve_grupo
    if se_ve is False and base[grupo] not in (reglas.no_se_ve_sin_defecto, NO_VISIBLE):
        etiquetas[grupo] = NO_VISIBLE
    return etiquetas


def leer_excepciones(
    ruta: str | Path, reglas: Reglas, fotos: list[Foto]
) -> tuple[list[Excepcion], list[str]]:
    """Lee el CSV de excepciones. Devuelve (excepciones válidas, errores).

    En lenguaje llano: cada línea fija la etiqueta de un grupo en una toma (``lechuga``,
    ``estado``, ``grupo``, ``etiqueta``) y, si se da ``vista`` (C, F, G o B), solo en esa vista.
    El estado puede ir como código o como el texto de la hoja. Se admite el BOM que pone Excel y
    el separador ``;`` de Excel en español. Es error un grupo o una etiqueta que no existen, una
    lechuga y estado sin fotos, una vista sin foto y dos líneas que fijan lo mismo (también una
    "todas las vistas" y otra de una vista concreta del mismo grupo).
    """
    ruta = Path(ruta)
    texto = ruta.read_text(encoding="utf-8-sig")  # utf-8, con o sin BOM
    primera = texto.splitlines()[0] if texto.strip() else ""
    separador = ";" if primera.count(";") > primera.count(",") else ","
    tabla = pd.read_csv(io.StringIO(texto), sep=separador, dtype=str, keep_default_na=False)
    tabla.columns = [str(c).strip() for c in tabla.columns]
    faltan = [c for c in _COLUMNAS_EXCEPCIONES if c not in tabla.columns]
    sobran = [c for c in tabla.columns if c not in (*_COLUMNAS_EXCEPCIONES, "vista")]
    if faltan or sobran:
        return [], [
            f"{ruta.name}: las columnas deben ser {', '.join(_COLUMNAS_EXCEPCIONES)} y, "
            f"opcional, vista (tiene: {', '.join(tabla.columns)})"
        ]

    tomas = {f.toma for f in fotos}
    existentes = {(f.lechuga, f.estado, f.letra) for f in fotos}
    excepciones, errores = [], []
    for posicion, fila in enumerate(tabla.to_dict("records")):
        linea = posicion + 2  # la línea 1 es la cabecera
        lechuga = fila["lechuga"].strip().upper()
        estado = traducir_estado(fila["estado"], reglas.alias) or fila["estado"].strip()
        grupo, etiqueta = fila["grupo"].strip(), fila["etiqueta"].strip()
        letra = (fila.get("vista") or "").strip().upper() or None
        problemas = []
        if (lechuga, estado) not in tomas:
            problemas.append(f"{lechuga}, estado {estado}: no hay fotos")
        if grupo not in GROUPS:
            problemas.append(f"grupo '{grupo}' no válido (válidos: {', '.join(GROUPS)})")
        elif etiqueta != NO_VISIBLE and etiqueta not in GROUPS[grupo]:
            problemas.append(
                f"etiqueta '{etiqueta}' no válida en '{grupo}' "
                f"(válidas: {', '.join((*GROUPS[grupo], NO_VISIBLE))})"
            )
        if letra is not None and letra not in reglas.vistas:
            problemas.append(f"vista '{letra}' no válida (válidas: {', '.join(reglas.vistas)})")
        elif letra is not None and (lechuga, estado) in tomas:
            if (lechuga, estado, letra) not in existentes:
                problemas.append(f"no hay foto {lechuga}_{estado}_{letra}")
        if problemas:
            errores += [f"{ruta.name}, línea {linea}: {problema}" for problema in problemas]
            continue
        excepciones.append(Excepcion(linea, lechuga, estado, grupo, etiqueta, letra))

    por_grupo = defaultdict(list)
    for excepcion in excepciones:
        por_grupo[(excepcion.lechuga, excepcion.estado, excepcion.grupo)].append(excepcion)
    for (lechuga, estado, grupo), lista in por_grupo.items():
        letras = [e.letra for e in lista]
        if len(lista) > 1 and (None in letras or len(set(letras)) < len(letras)):
            errores.append(
                f"{ruta.name}, líneas {' y '.join(str(e.linea) for e in lista)}: excepción "
                f"repetida o solapada para {lechuga}, estado {estado}, grupo {grupo}"
            )
    return excepciones, errores


def aplicar_excepciones(
    etiquetas: dict[str, str], foto: Foto, excepciones: list[Excepcion]
) -> dict[str, str]:
    """Etiquetas de la foto con las excepciones que le tocan aplicadas (ganan sobre todo).

    Una excepción solo toca lo que nombra: su lechuga y estado y, si la da, su vista. No se
    propaga a otros estados (la herencia del estado 0 usa la regla por variedad, no la foto).
    """
    resultado = dict(etiquetas)
    for excepcion in excepciones:
        if (excepcion.lechuga, excepcion.estado) == foto.toma and excepcion.letra in (
            None,
            foto.letra,
        ):
            resultado[excepcion.grupo] = excepcion.etiqueta
    return resultado


# --- metadata.csv -------------------------------------------------------------------------


def _numero_texto(numero: float) -> str:
    """Número como texto, sin ``.0`` si es entero (24 y no 24.0)."""
    return str(int(numero)) if float(numero).is_integer() else str(numero)


def filas_metadata(
    fotos: list[Foto],
    filas: dict[tuple[str, str], FilaHoja],
    etiquetas: dict[str, dict[str, str]],
    reglas: Reglas,
    tabla: dict,
) -> pd.DataFrame:
    """Una fila de ``metadata.csv`` por foto, con todas las columnas del esquema como texto.

    En lenguaje llano: identidad de la foto (``image_id``, lechuga, vista, fuente, licencia,
    sesión), las seis etiquetas y la ``salida_global`` que sale de la regla de agregación
    existente, y el contexto de captura: variedad, ``estado_seq`` (el código del estado; los de la
    serie temporal valen 0), ``horas_desde_corte`` (solo en la serie), conservación, intervención
    (texto del estado de la hoja y las notas), dispositivo y nombre del archivo original.
    ``filas`` es ``{toma: FilaHoja}`` y ``etiquetas`` es ``{image_id: {grupo: etiqueta}}``.
    """
    registros = []
    for foto in fotos:
        hoja = filas[foto.toma]
        etiquetas_foto = etiquetas[foto.image_id]
        en_serie = foto.estado in reglas.serie
        registro = dict.fromkeys(METADATA_COLUMNS, "")
        registro.update(etiquetas_foto)
        registro.update(
            image_id=foto.image_id,
            fuente=reglas.fuente,
            licencia=reglas.licencia,
            etiqueta_original=hoja.estado_texto,
            lechuga_id=foto.lechuga,
            vista=reglas.vistas[foto.letra],
            sesion=reglas.sesion,
            salida_global=aggregate(etiquetas_foto, tabla).level,
            variedad=hoja.variedad,
            estado_seq="0" if en_serie else foto.estado,
            horas_desde_corte=_numero_texto(reglas.serie[foto.estado]) if en_serie else "",
            conservacion=reglas.conservacion[foto.lechuga],
            intervencion=(
                f"{hoja.estado_texto}; notas: {hoja.notas}" if hoja.notas else hoja.estado_texto
            ),
            dispositivo=reglas.dispositivo,
            archivo_original=foto.nombre,
        )
        registros.append(registro)
    return pd.DataFrame(registros, columns=list(METADATA_COLUMNS))


def fusionar_sesion(
    existente: pd.DataFrame | None, nuevas: pd.DataFrame, fuente: str, sesion: str
) -> pd.DataFrame:
    """Quita de la tabla existente las filas de esa fuente Y sesión y añade las nuevas al final.

    En lenguaje llano: repetir la ingesta de una sesión sustituye sus filas (no las duplica) y
    deja intactas las de otras fuentes y las de otras sesiones de la misma fuente. No modifica
    las tablas de entrada.
    """
    if existente is None or existente.empty:
        return nuevas.copy()
    faltan = [c for c in ("fuente", "sesion") if c not in existente.columns]
    if faltan:
        raise ValueError(f"El metadata.csv existente no tiene la columna {', '.join(faltan)}")
    de_esta = (existente["fuente"] == fuente) & (existente["sesion"] == sesion)
    return pd.concat([existente[~de_esta], nuevas], ignore_index=True)


def _avisos_otras_sesiones(
    existente: pd.DataFrame | None, reglas: Reglas, lechugas: set[str]
) -> list[str]:
    """Avisa si una lechuga de esta ingesta ya está en otra sesión de la misma fuente."""
    if existente is None or not {"fuente", "sesion", "lechuga_id"} <= set(existente.columns):
        return []
    otras = existente[
        (existente["fuente"] == reglas.fuente)
        & (existente["sesion"] != reglas.sesion)
        & existente["lechuga_id"].isin(lechugas)
    ]
    avisos = []
    for lechuga, filas in otras.groupby("lechuga_id", sort=True):
        avisos.append(
            f"{lechuga} ya está en metadata.csv en otra sesión de '{reglas.fuente}' "
            f"({', '.join(sorted(set(filas['sesion'])))}): si es otra lechuga, usa otro número "
            "para que la partición por lechuga no las junte"
        )
    return avisos


# --- Salida -------------------------------------------------------------------------------


def archivos_de_salida(salida: Path, patron: re.Pattern) -> list[Path]:
    """Archivos que dejó una ingesta anterior en la carpeta de salida (los que se pueden borrar).

    En lenguaje llano: como la ingesta borra la salida anterior, solo se atreve si la carpeta no
    existe o si solo contiene lo que ella misma escribe (jpg con nombre de foto e
    ``informe.json``). Si hay cualquier otra cosa lanza ``ValueError`` y no toca nada.
    """
    salida = Path(salida)
    if not salida.exists():
        return []
    if not salida.is_dir():
        raise ValueError(f"La salida {salida.name} existe y no es una carpeta")
    propios, ajenos = [], []
    for ruta in sorted(salida.iterdir()):
        es_foto = ruta.suffix == ".jpg" and patron.fullmatch(ruta.stem)
        if ruta.is_file() and (ruta.name == INFORME or es_foto):
            propios.append(ruta)
        else:
            ajenos.append(ruta.name)
    if ajenos:
        raise ValueError(
            f"No se toca la salida {salida.name}: contiene archivos que no son de la ingesta "
            f"({', '.join(ajenos)}). Muévelos o elige otra carpeta de salida."
        )
    return propios


def exportar_fotos(fotos: list[Foto], origen: Path, salida: Path) -> None:
    """Escribe cada foto como ``<salida>/<image_id>.jpg``, girada según su EXIF y sin metadatos.

    En lenguaje llano: ``leer_imagen`` decodifica con OpenCV, que aplica la orientación del EXIF
    (la foto queda como se ve en el móvil), y ``guardar_imagen`` codifica de nuevo con OpenCV,
    que no escribe EXIF ni otros metadatos (GPS, modelo...). Se vuelve a comprimir en JPEG con la
    calidad por defecto de OpenCV (95). Una prueba comprueba las dos cosas por si cambia OpenCV.
    """
    salida.mkdir(parents=True, exist_ok=True)
    for foto in fotos:
        guardar_imagen(salida / f"{foto.image_id}.jpg", leer_imagen(origen / foto.nombre))


# --- Informe ------------------------------------------------------------------------------


def _relativa(ruta: str | Path, raiz: Path) -> str:
    """Ruta para mostrar: relativa a la raíz si está dentro; si no, solo el nombre."""
    ruta = Path(ruta)
    if not ruta.is_absolute():
        return ruta.as_posix()
    try:
        return ruta.resolve().relative_to(raiz.resolve()).as_posix()
    except ValueError:
        return ruta.name


def construir_informe(
    *,
    config: str,
    salida: str,
    reglas: Reglas,
    cruce: Cruce,
    ignoradas: list[dict],
    desconocidos: list[dict],
    no_sin_defecto: list[dict],
    excepciones: list[Excepcion],
    etiquetas: dict[str, dict[str, str]],
    avisos: list[str],
    tabla: dict,
) -> dict:
    """Resumen de la ingesta, listo para guardar como JSON (sin rutas absolutas).

    Incluye fotos y lechugas, las fotos de cada lechuga y estado, las tomas con vistas que
    faltan, las fotos sin fila (siempre vacío: si las hay, la ingesta para con error), las filas
    sin fotos, las ignoradas por guiones, los recuentos desconocidos, los "No" que no cambian
    nada por no haber defecto, el número de excepciones, la distribución de etiquetas por grupo
    (todas, también las que salen 0 veces) y de la salida global, y los avisos generales.
    """
    distribucion = {g: dict.fromkeys((*GROUPS[g], NO_VISIBLE), 0) for g in GROUPS}
    niveles = dict.fromkeys(GLOBAL_LEVELS, 0)
    for etiquetas_foto in etiquetas.values():
        for grupo, etiqueta in etiquetas_foto.items():
            distribucion[grupo][etiqueta] += 1
        niveles[aggregate(etiquetas_foto, tabla).level] += 1
    return {
        "fuente": reglas.fuente,
        "sesion": reglas.sesion,
        "config": config,
        "salida": salida,
        "fotos": len(etiquetas),
        "lechugas": len({lechuga for lechuga, _ in cruce.tomas}),
        "fotos_por_lechuga_estado": [
            {
                "lechuga": lechuga,
                "estado": estado,
                "fotos": len(fotos),
                "vistas": [f.letra for f in fotos],
            }
            for (lechuga, estado), fotos in cruce.tomas.items()
        ],  # fmt: skip
        "vistas_incompletas": cruce.incompletas,
        "fotos_sin_fila": [],
        "filas_sin_fotos": cruce.filas_sin_fotos,
        "filas_ignoradas": ignoradas,
        "recuentos_desconocidos": desconocidos,
        "no_sin_defecto": no_sin_defecto,
        "excepciones": len(excepciones),
        "etiquetas": distribucion,
        "salida_global": niveles,
        "avisos": list(avisos),
    }


def formatear_informe(informe: dict) -> str:
    """Texto legible del informe para la consola."""
    lineas = [
        f"Ingesta de fotos propias (fuente {informe['fuente']}, sesión {informe['sesion']})",
        f"  fotos: {informe['fotos']} de {informe['lechugas']} lechugas",
        "  fotos por lechuga y estado:",
    ]
    por_lechuga = defaultdict(list)
    for toma in informe["fotos_por_lechuga_estado"]:
        por_lechuga[toma["lechuga"]].append(f"{toma['estado']} ({toma['fotos']})")
    lineas += [f"    {lechuga}: {', '.join(tomas)}" for lechuga, tomas in por_lechuga.items()]
    lineas.append(f"  filas de la hoja ignoradas por guiones: {len(informe['filas_ignoradas'])}")
    lineas.append(f"  excepciones: {informe['excepciones']}")
    lineas.append("  etiquetas por grupo (nº de fotos):")
    for grupo, cuentas in informe["etiquetas"].items():
        lineas.append(f"    {grupo}: {', '.join(f'{e} {n}' for e, n in cuentas.items())}")
    lineas.append(
        "  salida_global: "
        + ", ".join(f"{nivel} {n}" for nivel, n in informe["salida_global"].items())
    )
    for toma in informe["vistas_incompletas"]:
        lineas.append(
            f"  AVISO: {toma['lechuga']}, estado {toma['estado']}: faltan las vistas "
            f"{', '.join(toma['faltan'])}"
        )
    for fila in informe["filas_sin_fotos"]:
        lineas.append(
            f"  AVISO: fila {fila['fila']} de la hoja ({fila['lechuga']}, estado "
            f"{fila['estado']}): no hay fotos"
        )
    for dato in informe["recuentos_desconocidos"]:
        lineas.append(
            f"  AVISO: {dato['lechuga']}, estado {dato['estado']}: recuento desconocido en "
            f"'{dato['columna']}', {dato['grupo']} queda en NO_VISIBLE"
        )
    for toma in informe["no_sin_defecto"]:
        lineas.append(
            f'  AVISO: {toma["lechuga"]}, estado {toma["estado"]}: la hoja dice "No" en '
            "¿se ve el daño?, pero no hay defecto que ocultar: no cambia nada"
        )
    lineas += [f"  AVISO: {aviso}" for aviso in informe["avisos"]]
    return "\n".join(lineas)


# --- Orquestación -------------------------------------------------------------------------


def _juntar(errores: list[str]) -> ValueError:
    """Un único ``ValueError`` con todos los errores, uno por línea."""
    detalle = "\n  - ".join(errores)
    return ValueError(
        f"La ingesta tiene {len(errores)} error(es) y no se ha escrito nada:\n  - {detalle}"
    )


def _fallar_si_hay(errores: list[str]) -> None:
    """Lanza el ``ValueError`` de ``_juntar`` si hay algún error."""
    if errores:
        raise _juntar(errores)


def ingerir(
    ruta_config: str | Path,
    *,
    raiz: str | Path,
    metadata: str | Path,
    tabla: dict,
    comprobar: bool = False,
) -> dict:
    """Ingiere una sesión de fotos propias: fotos limpias, ``informe.json`` y ``metadata.csv``.

    En lenguaje llano, en este orden:

    1. lee el YAML, las fotos, la hoja y (si las hay) las excepciones, y los cruza;
    2. si hay errores, los junta en un único ``ValueError`` y no escribe nada;
    3. etiqueta cada foto, construye sus filas, las fusiona con ``metadata.csv`` y valida la
       tabla completa con el esquema; comprueba también que la carpeta de salida se puede borrar;
    4. con ``comprobar=True`` se para aquí y devuelve el informe sin escribir nada;
    5. si no, borra la salida anterior, escribe las fotos, luego ``informe.json`` y, por
       último, ``metadata.csv``. Así un fallo a mitad nunca deja el CSV a medias.

    Devuelve el informe. ``tabla`` es la de ``configs/salida_global.yaml``.
    """
    raiz = Path(raiz)
    config = cargar_config(ruta_config, raiz)
    reglas = config.reglas
    avisos = []
    if not reglas.dispositivo:
        avisos.append("'dispositivo' está vacío en el YAML: la columna dispositivo irá vacía")

    fotos, errores = leer_fotos(config.fotos, reglas)
    try:
        crudas = leer_hoja(config.hoja, config.hoja_nombre, reglas.columnas)
    except ValueError as error:  # sin la hoja o sin su cabecera no hay nada que cruzar
        raise _juntar([*errores, str(error)]) from error
    filas, ignoradas, errores_hoja = filas_de_hoja(crudas, reglas)
    errores += errores_hoja
    cruce = cruzar(fotos, filas, ignoradas, reglas)
    errores += cruce.errores
    errores += [
        f"{lechuga}: no tiene conservacion en el YAML (añádela a 'conservacion')"
        for lechuga in sorted({f.lechuga for f in fotos} - set(reglas.conservacion))
    ]
    excepciones: list[Excepcion] = []
    if config.excepciones is not None:
        excepciones, errores_excepciones = leer_excepciones(config.excepciones, reglas, fotos)
        errores += errores_excepciones
    _fallar_si_hay(errores)

    # Etiquetas: primero por toma (estado, variedad y recuentos) y luego por foto.
    por_toma = {(f.lechuga, f.estado): f for f in filas}
    bases, desconocidos, no_sin_defecto = {}, [], []
    for toma in cruce.tomas:
        fila = por_toma[toma]
        bases[toma], grupos_desconocidos = etiquetas_de_estado(fila, reglas)
        for grupo in grupos_desconocidos:
            columna = reglas.columnas[reglas.recuentos[grupo][0]]
            desconocidos.append(
                {"lechuga": toma[0], "estado": toma[1], "grupo": grupo, "columna": columna}
            )
        sin_defecto = bases[toma][reglas.no_se_ve_grupo] == reglas.no_se_ve_sin_defecto
        if fila.se_ve is False and sin_defecto:  # el "No" no oculta nada: se anota
            no_sin_defecto.append({"lechuga": toma[0], "estado": toma[1]})
    etiquetas = {}
    for foto in fotos:
        de_foto = etiquetas_de_foto(bases[foto.toma], foto.letra, por_toma[foto.toma].se_ve, reglas)
        etiquetas[foto.image_id] = aplicar_excepciones(de_foto, foto, excepciones)

    nuevas = filas_metadata(fotos, por_toma, etiquetas, reglas, tabla)
    existente = leer_metadata(metadata)
    avisos += _avisos_otras_sesiones(existente, reglas, {f.lechuga for f in fotos})
    total = fusionar_sesion(existente, nuevas, reglas.fuente, reglas.sesion)
    comprobar_metadata(total)
    anteriores = archivos_de_salida(config.salida, reglas.patron_nombre)

    informe = construir_informe(
        config=_relativa(ruta_config, raiz),
        salida=_relativa(config.salida, raiz),
        reglas=reglas,
        cruce=cruce,
        ignoradas=ignoradas,
        desconocidos=desconocidos,
        no_sin_defecto=no_sin_defecto,
        excepciones=excepciones,
        etiquetas=etiquetas,
        avisos=avisos,
        tabla=tabla,
    )
    if comprobar:
        return informe

    for ruta in anteriores:
        ruta.unlink()
    exportar_fotos(fotos, config.fotos, config.salida)
    (config.salida / INFORME).write_text(
        json.dumps(informe, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    escribir_metadata(total, metadata)
    return informe
