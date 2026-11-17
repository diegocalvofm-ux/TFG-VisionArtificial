"""ATTRIBUTION.md: atribución de los datasets públicos, generada desde configs/datasets/.

Uso (desde la raíz del repositorio, en PowerShell):
    python -m src.data.atribucion              # reescribe ATTRIBUTION.md
    python -m src.data.atribucion --comprobar  # solo comprueba que está al día (código 1 si no)

En lenguaje llano: cada dataset público tiene su YAML descriptor en ``configs/datasets/`` (los que
tienen ``zip``). De ahí salen el nombre, el autor, la URL, la licencia y la fecha de descarga, así
que hay una sola fuente de verdad: para cambiar la atribución se cambia el YAML y se regenera el
archivo. Un campo que falta (``null``) sale como PENDIENTE. Una prueba comprueba que el archivo
del repositorio coincide con lo que generan los YAML.
"""

import argparse
import datetime
import sys
from pathlib import Path

from src.config import load_config

RAIZ = Path(__file__).resolve().parents[2]
PENDIENTE = "PENDIENTE"
# Campos de cada fila de la tabla, en orden (la licencia es obligatoria; el resto puede faltar).
CAMPOS = ("nombre", "autor", "url", "licencia", "fecha_descarga")

_CABECERA = """\
# Atribución de los datasets públicos

Este proyecto usa los datasets públicos de esta tabla. Sus imágenes y anotaciones no están en el
repositorio: se descargan aparte a `data/raw/`, fuera del control de versiones.

En el proyecto, sus imágenes se recortan por caja, se redimensionan y se reetiquetan con los
grupos de calidad propios del TFG; las anotaciones originales se traducen con las reglas de
`configs/datasets/`.

- **Dataset**: nombre del proyecto en Roboflow, según el `README.roboflow.txt` del ZIP.
- **Fecha de descarga**: la de exportación que indica ese mismo README.
- **PENDIENTE**: dato aún no confirmado.

Archivo generado con `python -m src.data.atribucion` a partir de los YAML de
`configs/datasets/`. No se edita a mano: se cambia el YAML y se vuelve a generar.
"""


def _texto(valor) -> str:
    """Valor de un campo como texto; vacío o ``null`` -> PENDIENTE."""
    if isinstance(valor, datetime.date):  # una fecha sin comillas en el YAML
        return valor.isoformat()
    if valor is None or not str(valor).strip():
        return PENDIENTE
    return str(valor).strip().replace("|", "\\|")  # una barra rompería la tabla de Markdown


def fuentes_publicas(carpeta: str | Path) -> list[dict]:
    """Datos de atribución de cada YAML con ``zip`` de la carpeta, ordenados por fuente.

    Error (``ValueError``): un dataset público sin ``fuente`` o sin ``licencia``.
    """
    fuentes = []
    for ruta in sorted(Path(carpeta).glob("*.yaml")):
        config = load_config(ruta)
        if "zip" not in config:  # no es el descriptor de un dataset público
            continue
        if not config.get("fuente"):
            raise ValueError(f"{ruta.name}: falta 'fuente'")
        if _texto(config.get("licencia")) == PENDIENTE:
            raise ValueError(f"{ruta.name}: falta la 'licencia' del dataset {config['fuente']}")
        fuentes.append({"fuente": str(config["fuente"]),
                        **{campo: _texto(config.get(campo)) for campo in CAMPOS}})  # fmt: skip
    return sorted(fuentes, key=lambda f: f["fuente"])


def generar(carpeta: str | Path) -> str:
    """Contenido de ATTRIBUTION.md: la cabecera fija y una fila por dataset público."""
    lineas = [
        _CABECERA,
        "| Fuente | Dataset | Autor | URL | Licencia | Fecha de descarga |",
        "|---|---|---|---|---|---|",
    ]
    for f in fuentes_publicas(carpeta):
        lineas.append(f"| {f['fuente']} | {f['nombre']} | {f['autor']} | {f['url']} | "
                      f"{f['licencia']} | {f['fecha_descarga']} |")  # fmt: skip
    return "\n".join(lineas) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Escribe ATTRIBUTION.md (o, con ``--comprobar``, mira si está al día). 0 = bien, 1 = no."""
    parser = argparse.ArgumentParser(
        prog="python -m src.data.atribucion",
        description="Genera ATTRIBUTION.md con los datasets públicos de configs/datasets/.",
    )
    parser.add_argument("--carpeta", type=Path, default=RAIZ / "configs" / "datasets")
    parser.add_argument("--salida", type=Path, default=RAIZ / "ATTRIBUTION.md")
    parser.add_argument("--comprobar", action="store_true", help="no escribe; solo compara")
    argumentos = parser.parse_args(argv)
    try:
        contenido = generar(argumentos.carpeta)
    except (ValueError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    if argumentos.comprobar:
        salida = argumentos.salida
        actual = salida.read_text(encoding="utf-8") if salida.is_file() else None
        if actual != contenido:
            print(f"{salida.name} no está al día: python -m src.data.atribucion", file=sys.stderr)
            return 1
        print(f"{salida.name} está al día")
        return 0
    argumentos.salida.write_text(contenido, encoding="utf-8", newline="\n")
    print(f"{argumentos.salida.name} escrito")
    return 0


if __name__ == "__main__":
    sys.exit(main())
