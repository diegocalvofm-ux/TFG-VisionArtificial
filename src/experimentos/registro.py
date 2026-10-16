"""Registro de experimentos en las notas de Obsidian (docs/experimentos/expNNN.md).

Uso (desde la raíz del repositorio, en PowerShell):
    python -m src.experimentos.registro nuevo --objetivo "..." --dataset "..."
    python -m src.experimentos.registro campos --exp exp001 --set estado=hecho --set valor=0.35
    python -m src.experimentos.registro ejecucion --exp exp001 --runs runs/v0_owlvit

Reglas: solo se escribe lo que se da por línea de comandos o lo que se lee en los JSON de
``runs/`` (detector, dispositivo, commit, árbol sucio, tamaño de imagen, memoria, carga del
modelo y tiempos). Nada se inventa: lo que falta queda vacío y se avisa de ello. Ninguna nota
ni mensaje lleva rutas absolutas ni el nombre de usuario del sistema.
"""

import argparse
import contextlib
import re
import sys
from datetime import date
from pathlib import Path

from src.experimentos.ejecuciones import (
    ResumenEjecucion,
    actualizar_filas,
    contenido_bloque,
    fila_ejecucion,
    parsear_tabla,
    resumir_carpeta,
)
from src.experimentos.nota import (
    CAMPOS_NO_EDITABLES,
    campos_de_plantilla,
    crear_nota,
    leer_bloque,
    leer_cabecera,
    leer_nota,
    modificar_nota,
    poner_bloque,
    poner_campos,
)
from src.experimentos.privacidad import mostrar, relativa, sanear, validar

# Raíz del repositorio: de aquí sale la carpeta docs/ por defecto.
RAIZ = Path(__file__).resolve().parents[2]

# Estados que usa el proyecto (brief de la v0): planificado -> hecho. Otro valor se acepta
# pero se avisa, por si es una errata.
ESTADOS = ("planificado", "hecho")

_ID_EXPERIMENTO = re.compile(r"^exp\d{3,}$")
_CAMPOS_DE_NUEVO = (
    "objetivo", "dataset", "dataset_version", "modelo", "config", "entorno", "python", "torch",
    "carpeta_drive", "commit",
)  # fmt: skip


def crear_parser() -> argparse.ArgumentParser:
    """Define las órdenes y los argumentos de la línea de comandos."""
    parser = argparse.ArgumentParser(
        prog="python -m src.experimentos.registro",
        description="Registra experimentos en las notas de Obsidian (docs/experimentos).",
    )
    parser.add_argument(
        "--docs", type=Path, default=RAIZ / "docs", help="bóveda de Obsidian (por defecto, docs/)"
    )
    ordenes = parser.add_subparsers(dest="orden", required=True)

    nuevo = ordenes.add_parser("nuevo", help="crea la siguiente nota desde la plantilla")
    for campo in _CAMPOS_DE_NUEVO:
        ayuda = None
        if campo == "config":
            ayuda = "archivo del YAML usado; se guarda su ruta relativa y una copia"
        nuevo.add_argument(f"--{campo.replace('_', '-')}", default=None, help=ayuda)

    campos = ordenes.add_parser("campos", help="cambia campos de la cabecera de una nota")
    campos.add_argument("--exp", required=True, help="id de la nota, p. ej. exp001")
    campos.add_argument(
        "--set", action="append", required=True, metavar="CLAVE=VALOR", help="repetible"
    )

    ejecucion = ordenes.add_parser(
        "ejecucion", help="anota una ejecución: todo se lee de los JSON de la carpeta de runs"
    )
    ejecucion.add_argument("--exp", required=True, help="id de la nota, p. ej. exp001")
    ejecucion.add_argument(
        "--runs", required=True, type=Path, help="carpeta con los JSON (dentro del repositorio)"
    )
    ejecucion.add_argument("--observacion", default=None, help="texto libre para la tabla")
    ejecucion.add_argument(
        "--commit",
        default=None,
        help="commit con el que se ejecutó, si los JSON no lo traen (p. ej. datos de Colab)",
    )
    return parser


class _Saneador:
    """Envoltorio de stdout/stderr que quita rutas absolutas y el usuario de todo lo que escribe.

    Es la red de seguridad de la privacidad: aunque un mensaje (o el texto de un error del sistema
    operativo) lleve una ruta completa, al imprimirse sale relativa o como ``<ruta>``.
    """

    def __init__(self, destino):
        self.destino = destino
        self.base: Path | None = None

    def write(self, texto: str) -> int:
        self.destino.write(sanear(texto, self.base))
        return len(texto)

    def flush(self) -> None:
        self.destino.flush()

    def __getattr__(self, nombre: str):  # isatty, encoding... los pide quien imprime
        return getattr(self.destino, nombre)


def _avisar(mensaje: str) -> None:
    print(f"Aviso: {mensaje}", file=sys.stderr)


def _ruta_nota(docs: Path, exp: str) -> Path:
    """Ruta de la nota ``exp``; ``ValueError`` si el id no es válido y ``FileNotFoundError``
    si la nota no existe."""
    if not _ID_EXPERIMENTO.match(exp):
        raise ValueError(f"id de experimento no válido: {exp!r} (debe ser como exp001)")
    ruta = docs / "experimentos" / f"{exp}.md"
    if not ruta.is_file():
        raise FileNotFoundError(f"No existe la nota {exp}")
    return ruta


def _campos_validos(docs: Path) -> list[str]:
    """Campos editables de la cabecera, tomados de la plantilla (la única fuente)."""
    plantilla = (docs / "plantillas" / "experimento.md").read_text(encoding="utf-8")
    return [c for c in campos_de_plantilla(plantilla) if c not in CAMPOS_NO_EDITABLES]


def _resolver(valor: Path, base: Path, tipo: str) -> Path:
    """Localiza un archivo (``tipo="archivo"``) o una carpeta (``"carpeta"``).

    En lenguaje llano: la ruta puede darse absoluta o relativa, desde donde se lanza el comando
    o desde la raíz del repositorio. ``FileNotFoundError`` si no existe.
    """
    ruta = Path(valor)
    candidatas = [ruta] if ruta.is_absolute() else [Path.cwd() / ruta, base / ruta]
    for candidata in candidatas:
        if candidata.is_file() if tipo == "archivo" else candidata.is_dir():
            return candidata
    if tipo == "archivo":
        raise FileNotFoundError(f"No existe el archivo de configuración: {valor}")
    raise FileNotFoundError(f"No existe la carpeta de resultados: {valor}")


def _nuevo(args: argparse.Namespace, hoy: date, base: Path) -> int:
    campos = {c: getattr(args, c) for c in _CAMPOS_DE_NUEVO if getattr(args, c) is not None}
    for clave, valor in campos.items():
        if clave != "config":  # la ruta de config se normaliza a relativa más abajo
            validar(clave, valor)

    despues = None
    if args.config is not None:
        ruta_config = _resolver(args.config, base, "archivo")
        campos["config"] = relativa(ruta_config, base)
        contenido = ruta_config.read_text(encoding="utf-8").rstrip()
        validar("config (copia del YAML)", contenido)
        copia = f"```yaml\n{contenido}\n```"

        def despues(texto: str) -> str:
            return poner_bloque(texto, "config", copia, "Configuración")

    carpeta = args.docs / "experimentos"
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta = crear_nota(carpeta, args.docs / "plantillas" / "experimento.md", campos, hoy, despues)

    print(f"Creada: {mostrar(ruta, base)}")
    vacios = [c for c, valor in leer_cabecera(leer_nota(ruta)).items() if valor is None]
    print(f"Faltan: {', '.join(vacios)}" if vacios else "No falta ningún campo.")
    return 0


def _campos(args: argparse.Namespace) -> int:
    ruta = _ruta_nota(args.docs, args.exp)
    cambios = {}
    for par in args.set:
        if "=" not in par:
            raise ValueError(f"--set espera clave=valor (recibido: {par!r})")
        clave, valor = par.split("=", 1)
        cambios[clave.strip()] = valor.strip()
    for clave, valor in cambios.items():
        validar(clave, valor)
    validos = _campos_validos(args.docs)

    modificar_nota(ruta, lambda texto: poner_campos(texto, cambios, validos))

    estado = cambios.get("estado")
    if estado is not None and estado not in ESTADOS:
        _avisar(f"el estado {estado!r} no es uno de los usados ({', '.join(ESTADOS)})")
    print(f"Actualizada {args.exp}: {', '.join(cambios)}")
    return 0


def _commit_a_anotar(args: argparse.Namespace, resumen: ResumenEjecucion, anotado) -> str | None:
    """Decide qué commit escribir en la cabecera (``None`` = dejarla como está).

    Prioridad: ``--commit`` y, si no, el ``env.commit`` de los JSON (si todos coinciden). No hay
    más fuentes: ya no se pregunta a git, porque el commit de ahora puede no ser el de la
    ejecución. Nunca se pisa un commit ya anotado (puede venir de Colab), solo se comenta.
    """
    if args.commit:
        return args.commit
    if resumen.commits_distintos or not resumen.commit:
        return None
    if anotado in (None, ""):
        return resumen.commit
    if str(anotado) != resumen.commit:
        print(
            f"La nota ya tiene el commit {anotado} y los JSON dicen {resumen.commit}; no se "
            "cambia (usa --commit para fijarlo)."
        )
    return None


def _avisos_de_la_ejecucion(
    args: argparse.Namespace, resumen: ResumenEjecucion, anotado
) -> None:
    """Imprime los avisos de una ejecución: árbol sucio, datos que faltan, JSON ignorados..."""
    if resumen.alguno_sucio:
        _avisar(
            "esa ejecución se hizo con cambios sin commitear en src o configs: "
            "no se reproduce solo con el commit"
        )
    for aviso in resumen.avisos:
        _avisar(aviso)
    if resumen.ignorados:
        _avisar(f"JSON ignorados por no tener el formato esperado: {', '.join(resumen.ignorados)}")
    if resumen.faltantes:
        total = resumen.n_imagenes
        detalle = ", ".join(f"{campo} ({n} de {total})" for campo, n in resumen.faltantes.items())
        _avisar(f"faltan datos en los JSON (las celdas quedan vacías): {detalle}")
    if anotado in (None, "") and not args.commit and not resumen.commit:
        if not resumen.commits_distintos:
            _avisar(
                "ningún JSON trae env.commit: el campo 'commit' de la cabecera queda vacío "
                "(usa --commit si lo conoces)"
            )


def _ejecucion(args: argparse.Namespace, base: Path) -> int:
    ruta = _ruta_nota(args.docs, args.exp)
    carpeta = _resolver(args.runs, base, "carpeta")
    carpeta_relativa = relativa(carpeta, base)
    resumen = resumir_carpeta(carpeta)

    fila = fila_ejecucion(carpeta_relativa, resumen)
    for columna, valor in fila.items():
        validar(columna, valor)
    for clave, valor in (("observacion", args.observacion), ("commit", args.commit)):
        if valor is not None:
            validar(clave, valor)
    validos = _campos_validos(args.docs)
    anotado = leer_cabecera(leer_nota(ruta)).get("commit")
    commit = _commit_a_anotar(args, resumen, anotado)

    def aplicar(texto: str) -> str:
        filas = parsear_tabla(leer_bloque(texto, "resultados") or "")
        filas = actualizar_filas(filas, fila, args.observacion)
        texto = poner_bloque(texto, "resultados", contenido_bloque(filas), "Resultados")
        return poner_campos(texto, {"commit": commit}, validos) if commit else texto

    modificar_nota(ruta, aplicar)

    _avisos_de_la_ejecucion(args, resumen, anotado)
    print(f"Registrada en {args.exp}: {carpeta_relativa}, {resumen.n_imagenes} imágenes.")
    return 0


def main(argv=None, *, hoy: date | None = None) -> int:
    """Punto de entrada. Devuelve ``0`` si fue bien y ``2`` si hubo un error de uso.

    En lenguaje llano: lee la orden, la ejecuta y convierte cualquier error previsto en un
    mensaje claro con código 2. Todo lo que se imprime pasa por ``sanear``, así que ninguna
    salida lleva rutas absolutas ni el usuario del sistema.

    ``hoy`` se puede sustituir para probarlo sin depender de la fecha.
    """
    hoy = hoy or date.today()
    salida, errores = _Saneador(sys.stdout), _Saneador(sys.stderr)
    with contextlib.redirect_stdout(salida), contextlib.redirect_stderr(errores):
        args = crear_parser().parse_args(argv)
        base = args.docs.resolve().parent  # la raíz del repo (la carpeta que contiene docs/)
        salida.base = errores.base = base
        try:
            if args.orden == "nuevo":
                return _nuevo(args, hoy, base)
            if args.orden == "campos":
                return _campos(args)
            return _ejecucion(args, base)
        except (OSError, ValueError) as error:  # incluye FileNotFoundError y FileExistsError
            print(f"Error: {error}", file=sys.stderr)
            return 2


if __name__ == "__main__":
    raise SystemExit(main())
