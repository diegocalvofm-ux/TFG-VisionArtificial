"""Resumen de una ejecución del pipeline y tabla de resultados de la nota del experimento.

Una "ejecución" es una carpeta con un JSON por imagen (los que escribe ``src.pipeline.run``).
Aquí solo se calcula lo que se puede MEDIR en esos JSON: no se estima ni se inventa nada. Desde
la tarea 5 cada JSON trae, además de los tiempos, ``env.detector``, ``env.device``,
``env.commit``, ``env.dirty``, ``image_size``, ``rss_mb``, ``model_load_ms`` y, en
``timing_ms``, ``repeticiones`` y ``calentamiento``. Si un JSON (por ejemplo, uno anterior a
la tarea 5) no trae alguno, esa celda queda vacía y el resumen cuenta en cuántos JSON falta,
para poder avisar.
"""

import json
import math
import re
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from src.data.schema import GLOBAL_LEVELS

# Columnas de la tabla de Resultados de la nota, en su orden. "Carpeta" identifica la fila: es
# la carpeta de runs (ruta relativa a la raíz del repo). Las columnas de niveles llevan el
# nombre del nivel.
COLUMNAS = (
    "Carpeta",
    "Detector",
    "Dispositivo",
    "Commit",
    "Sucio",
    "Imágenes",
    "Tamaño imagen",
    "Repeticiones",
    "Calentamiento",
    "Mediana (ms)",
    "Mediana detección (ms)",
    "Carga modelo (ms)",
    "Memoria (MB)",
    "Lechugas",
    "Sin detección",
    *GLOBAL_LEVELS,
    "Observación",
)

NOTA_MEDIANA = (
    "Mediana (ms) y Mediana detección (ms): mediana entre imágenes de `timing_ms.total` y "
    "`timing_ms.detect` de cada JSON; cada valor del JSON ya es la mediana de `Repeticiones` "
    "pasadas medidas tras `Calentamiento` pasadas de calentamiento. Las celdas vacías son datos "
    "que el JSON no traía (los JSON anteriores a la tarea 5 no los llevan): no se rellenan a mano."
)

_PIPE_SIN_ESCAPAR = re.compile(r"(?<!\\)\|")
_SEPARADOR = re.compile(r"^:?-+:?$")
_VARIOS = "varios"


@dataclass(frozen=True)
class ResumenEjecucion:
    """Lo medido en una carpeta de JSON. Los campos opcionales valen ``None`` si no se tienen.

    ``faltantes`` dice, por dato del JSON, en cuántos JSON válidos no está (para avisar).
    ``alguno_sucio`` es ``True`` si algún JSON declara ``dirty: true``, aunque otros no.
    """

    n_imagenes: int
    ignorados: tuple[str, ...]
    mediana_total_ms: float
    mediana_detect_ms: float
    lechugas: int
    sin_detecciones: int
    niveles: dict[str, int]
    detector: str | None = None
    dispositivo: str | None = None
    commit: str | None = None
    sucio: bool | None = None
    tamano_imagen: str | None = None
    repeticiones: int | str | None = None
    calentamiento: int | str | None = None
    memoria_max_mb: float | None = None
    carga_modelo_ms: float | None = None
    commits_distintos: bool = False
    alguno_sucio: bool = False
    avisos: tuple[str, ...] = ()
    faltantes: dict[str, int] = field(default_factory=dict)


# --- Lectura de un JSON -------------------------------------------------------------------


def _numero(valor) -> float:
    """Un número válido (no booleano, finito y no negativo). Si no, ``ValueError``."""
    if isinstance(valor, bool) or not isinstance(valor, int | float):
        raise ValueError("no es un número")
    if not math.isfinite(valor) or valor < 0:
        raise ValueError("no es un valor válido")
    return valor


def _texto(valor) -> str:
    if not isinstance(valor, str) or not valor.strip():
        raise ValueError("no es un texto")
    return valor.strip()


def _booleano(valor) -> bool:
    if not isinstance(valor, bool):
        raise ValueError("no es verdadero/falso")
    return valor


def _entero_desde(minimo: int) -> Callable:
    """Validador de un entero (no booleano) mayor o igual que ``minimo``."""

    def validar(valor) -> int:
        if isinstance(valor, bool) or not isinstance(valor, int) or valor < minimo:
            raise ValueError(f"no es un entero >= {minimo}")
        return valor

    return validar


def _tamano(valor) -> str:
    """Tamaño de la imagen tal como lo trae el JSON: ``[a, b]`` pasa a ``axb``, en ESE orden."""
    if isinstance(valor, str):
        return _texto(valor)
    if isinstance(valor, list | tuple) and len(valor) == 2:
        lados = [_numero(lado) for lado in valor]
        if all(lado > 0 for lado in lados):
            return "x".join(str(int(lado) if float(lado).is_integer() else lado) for lado in lados)
    raise ValueError("no es un tamaño")


# Dato del JSON -> (nombre con el que se avisa, validador, ¿un ``null`` presente es normal?).
# Un ``model_load_ms: null`` es normal (detector sin modelo); los demás ``null`` son un dato
# que no se pudo obtener (p. ej. ``commit`` sin git) y cuentan como ausentes.
_OPCIONALES: dict[str, tuple[str, Callable, bool]] = {
    "detector": ("env.detector", _texto, False),
    "device": ("env.device", _texto, False),
    "commit": ("env.commit", _texto, False),
    "dirty": ("env.dirty", _booleano, False),
    "image_size": ("image_size", _tamano, False),
    "rss_mb": ("rss_mb", _numero, False),
    "model_load_ms": ("model_load_ms", _numero, True),
    "repeticiones": ("timing_ms.repeticiones", _entero_desde(1), False),
    "calentamiento": ("timing_ms.calentamiento", _entero_desde(0), False),
}


def _buscar(datos: dict, clave: str) -> tuple[bool, object]:
    """(¿está?, valor) de ``clave`` en el nivel superior del JSON, en ``env`` o en ``timing_ms``."""
    for lugar in (datos, datos.get("env"), datos.get("timing_ms")):
        if isinstance(lugar, dict) and clave in lugar:
            return True, lugar[clave]
    return False, None


def _leer_opcionales(datos: dict, nombre: str) -> tuple[dict, list[str], list[str]]:
    """Datos opcionales de un JSON: (válidos, avisos de formato, nombres de los que faltan).

    Un dato con formato incorrecto se ignora con un aviso propio (no se cuenta como ausente) y
    no invalida el JSON.
    """
    encontrados: dict = {}
    avisos: list[str] = []
    ausentes: list[str] = []
    for clave, (etiqueta, validar, nulo_normal) in _OPCIONALES.items():
        presente, valor = _buscar(datos, clave)
        if not presente or (valor is None and not nulo_normal):
            ausentes.append(etiqueta)
            continue
        if valor is None:
            continue
        try:
            encontrados[clave] = validar(valor)
        except ValueError:
            avisos.append(
                f"{nombre}: '{clave}' no tiene el formato esperado ({valor!r}); se ignora"
            )
    return encontrados, avisos, ausentes


def _leer_resultado(ruta: Path) -> tuple[float, float, list[str], dict, list[str], list[str]]:
    """Lee un JSON del pipeline.

    Devuelve (total_ms, detect_ms, niveles de sus lechugas, datos opcionales, avisos de
    formato, nombres de los datos que faltan).

    Error (ValueError): no es JSON, o le faltan los tiempos, o algún nivel no existe.
    """
    datos = json.loads(ruta.read_text(encoding="utf-8"))
    if not isinstance(datos, dict):
        raise ValueError("no es un objeto JSON")
    tiempos = datos.get("timing_ms")
    lechugas = datos.get("lettuces")
    if not isinstance(tiempos, dict) or not isinstance(lechugas, list):
        raise ValueError("faltan timing_ms o lettuces")
    niveles = []
    for lechuga in lechugas:
        nivel = lechuga.get("global") if isinstance(lechuga, dict) else None
        if nivel not in GLOBAL_LEVELS:
            raise ValueError(f"nivel global desconocido: {nivel!r}")
        niveles.append(nivel)
    total, detect = _numero(tiempos.get("total")), _numero(tiempos.get("detect"))
    opcionales, avisos, ausentes = _leer_opcionales(datos, ruta.name)
    return total, detect, niveles, opcionales, avisos, ausentes


def _unico(valores: list, clave: str, avisos: list[str]):
    """El valor si todos los JSON coinciden; ``None`` (con aviso) si se contradicen."""
    distintos = sorted({str(v) for v in valores})
    if len(distintos) > 1:
        avisos.append(f"los JSON tienen valores distintos de '{clave}': {', '.join(distintos)}")
        return None
    return valores[0] if valores else None


def _unico_o_varios(valores: list):
    """El valor si todos coinciden, ``"varios"`` si no, y ``None`` si no hay ninguno."""
    distintos = set(valores)
    if not distintos:
        return None
    return distintos.pop() if len(distintos) == 1 else _VARIOS


# --- Resumen de una carpeta ---------------------------------------------------------------


def resumir_carpeta(carpeta: Path) -> ResumenEjecucion:
    """Resume los ``*.json`` de una carpeta: nº de imágenes, medianas, lechugas y niveles.

    En lenguaje llano: abre cada JSON, apunta cuánto tardó y cuántas lechugas encontró y de
    qué nivel, y calcula las medianas entre imágenes (el valor del medio, que no se deja
    engañar por una imagen muy lenta). Los JSON que no tienen el formato esperado no se usan,
    pero se devuelven en ``ignorados`` para poder avisar de ellos. No se lee dentro de
    subcarpetas.

    También recoge lo que escribe el pipeline desde la tarea 5: memoria = el máximo de
    ``rss_mb``, carga del modelo = la mediana de ``model_load_ms``; detector, dispositivo,
    commit y ``dirty`` solo si todos los JSON coinciden (si se contradicen, se avisa y no se
    elige ninguno); y ``repeticiones``, ``calentamiento`` y tamaño de imagen (``"varios"`` si
    difieren). ``faltantes`` cuenta en cuántos JSON no está cada dato.

    Errores: ``FileNotFoundError`` si la carpeta no existe; ``ValueError`` si no hay ningún
    JSON válido.
    """
    carpeta = Path(carpeta)
    if not carpeta.is_dir():
        raise FileNotFoundError(f"No existe la carpeta de resultados: {carpeta}")

    totales, detecciones, ignorados, avisos = [], [], [], []
    por_campo: dict[str, list] = {clave: [] for clave in _OPCIONALES}
    cuenta_faltantes: dict[str, int] = {}
    niveles = dict.fromkeys(GLOBAL_LEVELS, 0)
    lechugas = sin_detecciones = 0
    for ruta in sorted(carpeta.glob("*.json"), key=lambda p: p.name):
        try:
            total, detect, de_la_imagen, opcionales, de_este_json, ausentes = _leer_resultado(ruta)
        except (OSError, ValueError):  # incluye JSON mal formado y UnicodeDecodeError
            ignorados.append(ruta.name)
            continue
        totales.append(total)
        detecciones.append(detect)
        avisos += de_este_json
        for etiqueta in ausentes:
            cuenta_faltantes[etiqueta] = cuenta_faltantes.get(etiqueta, 0) + 1
        for clave, valor in opcionales.items():
            por_campo[clave].append(valor)
        lechugas += len(de_la_imagen)
        sin_detecciones += not de_la_imagen
        for nivel in de_la_imagen:
            niveles[nivel] += 1

    if not totales:
        extra = f" ({len(ignorados)} ignorados: {', '.join(ignorados)})" if ignorados else ""
        raise ValueError(f"No hay JSON válidos del pipeline en {carpeta}{extra}")

    commits = por_campo["commit"]
    # En el orden de _OPCIONALES, para que el aviso salga siempre igual.
    faltantes = {
        etiqueta: cuenta_faltantes[etiqueta]
        for etiqueta, _, _ in _OPCIONALES.values()
        if etiqueta in cuenta_faltantes
    }
    return ResumenEjecucion(
        n_imagenes=len(totales),
        ignorados=tuple(ignorados),
        mediana_total_ms=statistics.median(totales),
        mediana_detect_ms=statistics.median(detecciones),
        lechugas=lechugas,
        sin_detecciones=sin_detecciones,
        niveles=niveles,
        detector=_unico(por_campo["detector"], "detector", avisos),
        dispositivo=_unico(por_campo["device"], "device", avisos),
        commit=_unico(commits, "commit", avisos),
        sucio=_unico(por_campo["dirty"], "dirty", avisos),
        tamano_imagen=_unico_o_varios(por_campo["image_size"]),
        repeticiones=_unico_o_varios(por_campo["repeticiones"]),
        calentamiento=_unico_o_varios(por_campo["calentamiento"]),
        memoria_max_mb=max(por_campo["rss_mb"]) if por_campo["rss_mb"] else None,
        carga_modelo_ms=(
            statistics.median(por_campo["model_load_ms"]) if por_campo["model_load_ms"] else None
        ),
        commits_distintos=len(set(commits)) > 1,
        alguno_sucio=any(por_campo["dirty"]),
        avisos=tuple(avisos),
        faltantes=faltantes,
    )


# --- La tabla de la nota ------------------------------------------------------------------


def _decimal(valor: float | None) -> str:
    return "" if valor is None else f"{valor:.1f}"


def _texto_o_vacio(valor) -> str:
    return "" if valor is None else str(valor)


def fila_ejecucion(carpeta: str, resumen: ResumenEjecucion) -> dict[str, str]:
    """Convierte un resumen en una fila de la tabla (todo como texto, en orden de COLUMNAS).

    ``carpeta`` es la ruta relativa de la carpeta de runs y es lo que identifica la fila. Lo
    que el JSON no trajo (detector, dispositivo, commit, sucio, tamaño, repeticiones,
    calentamiento, carga y memoria) queda vacío, nunca a 0. La observación también (la da el
    autor).
    """
    sucio = "" if resumen.sucio is None else ("sí" if resumen.sucio else "no")
    fila = {
        "Carpeta": carpeta,
        "Detector": resumen.detector or "",
        "Dispositivo": resumen.dispositivo or "",
        "Commit": resumen.commit or "",
        "Sucio": sucio,
        "Imágenes": str(resumen.n_imagenes),
        "Tamaño imagen": resumen.tamano_imagen or "",
        "Repeticiones": _texto_o_vacio(resumen.repeticiones),
        "Calentamiento": _texto_o_vacio(resumen.calentamiento),
        # Dos decimales: un detector trivial tarda décimas o centésimas de milisegundo.
        "Mediana (ms)": f"{resumen.mediana_total_ms:.2f}",
        "Mediana detección (ms)": f"{resumen.mediana_detect_ms:.2f}",
        "Carga modelo (ms)": _decimal(resumen.carga_modelo_ms),
        "Memoria (MB)": _decimal(resumen.memoria_max_mb),
        "Lechugas": str(resumen.lechugas),
        "Sin detección": str(resumen.sin_detecciones),
    }
    fila.update({nivel: str(resumen.niveles.get(nivel, 0)) for nivel in GLOBAL_LEVELS})
    fila["Observación"] = ""
    return fila


def escapar_celda(valor) -> str:
    """Texto seguro para una celda de tabla Markdown: sin saltos de línea y con ``|`` escapado."""
    return str(valor).replace("\n", " ").replace("|", r"\|").strip()


def renderizar_tabla(filas: Sequence[dict[str, str]]) -> str:
    """Escribe las filas como tabla Markdown con las columnas de COLUMNAS."""
    cabecera = "| " + " | ".join(COLUMNAS) + " |"
    separador = "| " + " | ".join("---" for _ in COLUMNAS) + " |"
    cuerpo = [
        "| " + " | ".join(escapar_celda(fila.get(columna, "")) for columna in COLUMNAS) + " |"
        for fila in filas
    ]
    return "\n".join([cabecera, separador, *cuerpo])


def _celdas(linea: str) -> list[str]:
    interior = linea.strip()[1:-1]
    return [c.strip().replace(r"\|", "|") for c in _PIPE_SIN_ESCAPAR.split(interior)]


def parsear_tabla(texto: str) -> list[dict[str, str]]:
    """Lee las filas de la primera tabla Markdown de un texto (lista vacía si no hay ninguna).

    Una tabla con menos columnas que las actuales (de una versión anterior) se lee igual: las
    columnas que no tenía quedan vacías al volver a escribirla.
    """
    lineas = [linea for linea in texto.split("\n") if linea.strip().startswith("|")]
    if len(lineas) < 2:
        return []
    nombres = _celdas(lineas[0])
    filas = []
    for linea in lineas[1:]:
        celdas = _celdas(linea)
        if all(_SEPARADOR.match(celda) for celda in celdas):
            continue  # la línea "| --- | --- |"
        celdas += [""] * (len(nombres) - len(celdas))
        filas.append(dict(zip(nombres, celdas, strict=False)))
    return filas


def actualizar_filas(
    filas: Sequence[dict[str, str]], nueva: dict[str, str], observacion: str | None
) -> list[dict[str, str]]:
    """Devuelve las filas con ``nueva`` añadida o, si ya había esa ejecución, sustituida.

    En lenguaje llano: una ejecución se identifica por su carpeta de runs. Registrar otra vez
    la misma carpeta actualiza su fila en lugar de duplicarla; dos carpetas distintas son dos
    filas aunque sean del mismo detector. Una fila sin carpeta (de una tabla anterior) nunca
    se confunde con otra. La observación que escribió el autor se conserva salvo que se dé una
    nueva. No modifica la lista recibida.
    """
    resultado = [dict(fila) for fila in filas]
    carpeta = nueva.get("Carpeta", "")
    for posicion, fila in enumerate(resultado):
        if carpeta and fila.get("Carpeta") == carpeta:
            previa = fila.get("Observación", "")
            resultado[posicion] = {
                **nueva,
                "Observación": previa if observacion is None else observacion,
            }
            return resultado
    resultado.append({**nueva, "Observación": observacion or nueva.get("Observación", "")})
    return resultado


def contenido_bloque(filas: Sequence[dict[str, str]]) -> str:
    """Texto del bloque automático de Resultados: una nota explicativa y la tabla."""
    return f"{NOTA_MEDIANA}\n\n{renderizar_tabla(filas)}"
