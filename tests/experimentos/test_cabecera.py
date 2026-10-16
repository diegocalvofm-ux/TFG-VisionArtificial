"""La cabecera de las notas es YAML válido y respeta las propiedades de la plantilla real.

Las pruebas leen la cabecera con ``yaml.safe_load`` (lo mismo que haría un lector externo) y no
con las funciones del propio sistema, para no darle por bueno lo que él mismo escribe.
"""

import shutil
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from src.experimentos.nota import campos_de_plantilla
from src.experimentos.registro import main

RAIZ = Path(__file__).resolve().parents[2]
PLANTILLA_REAL = RAIZ / "docs" / "plantillas" / "experimento.md"
PROPIEDADES = [
    "id", "fecha", "estado", "objetivo", "dataset", "dataset_version", "modelo", "config",
    "commit", "entorno", "python", "torch", "metrica_principal", "valor", "carpeta_drive", "tags",
]  # fmt: skip
EJECUCION = (
    "ejecucion", "--exp", "exp001", "--runs"
)  # fmt: skip
INICIO = "<!-- registro:resultados -->"
FIN = "<!-- /registro:resultados -->"


@pytest.fixture
def boveda(tmp_path):
    """Una bóveda de pruebas con la plantilla REAL del repositorio (solo se lee)."""
    raiz = tmp_path / "docs"
    (raiz / "plantillas").mkdir(parents=True)
    (raiz / "experimentos").mkdir()
    shutil.copy(PLANTILLA_REAL, raiz / "plantillas" / "experimento.md")
    return SimpleNamespace(raiz=raiz, carpeta=raiz / "experimentos")


def registrar(boveda, *args):
    return main(["--docs", str(boveda.raiz), *args], hoy=date(2026, 10, 7))


def texto(boveda, exp="exp001"):
    return (boveda.carpeta / f"{exp}.md").read_text(encoding="utf-8")


def cabecera(boveda, exp="exp001"):
    """La cabecera leída con yaml.safe_load, sin pasar por el código que se prueba."""
    lineas = texto(boveda, exp).split("\n")
    assert lineas[0] == "---"
    cierre = lineas.index("---", 1)
    datos = yaml.safe_load("\n".join(lineas[1:cierre]))
    assert isinstance(datos, dict)
    return datos


def _runs(tmp_path, escribir_json, n=2):
    for i in range(n):
        escribir_json(tmp_path / "runs", f"f{i}", total_ms=10 * (i + 1), completo=True)
    return tmp_path / "runs"


# --- nombres de propiedades ---------------------------------------------------------------


def test_la_plantilla_real_tiene_exactamente_las_propiedades_acordadas():
    texto_plantilla = PLANTILLA_REAL.read_text(encoding="utf-8")

    assert list(campos_de_plantilla(texto_plantilla)) == PROPIEDADES


def test_una_nota_nueva_tiene_las_propiedades_de_la_plantilla_y_en_su_orden(boveda):
    registrar(boveda, "nuevo")

    assert list(cabecera(boveda)) == PROPIEDADES


def test_los_nombres_siguen_siendo_los_mismos_tras_todas_las_ordenes(
    boveda, escribir_json, tmp_path
):
    registrar(boveda, "nuevo", "--objetivo", "o")
    registrar(boveda, *EJECUCION, str(_runs(tmp_path, escribir_json)))
    registrar(boveda, "campos", "--exp", "exp001", "--set", "estado=hecho",
              "--set", "valor=0.35", "--set", "metrica_principal=mAP50")  # fmt: skip

    assert list(cabecera(boveda)) == PROPIEDADES


# --- tipos --------------------------------------------------------------------------------


def test_los_tipos_de_una_nota_nueva(boveda):
    registrar(boveda, "nuevo")

    datos = cabecera(boveda)

    assert datos["id"] == "exp001" and isinstance(datos["id"], str)
    assert datos["fecha"] == date(2026, 10, 7)
    assert datos["estado"] == "planificado"
    assert datos["tags"] == ["experimento"]  # una lista, no un texto
    sin_dato = set(PROPIEDADES) - {"id", "fecha", "estado", "tags"}
    assert all(datos[c] is None for c in sin_dato)


def test_los_campos_de_texto_siguen_siendo_texto_aunque_parezcan_numeros(boveda):
    registrar(
        boveda, "nuevo", "--dataset-version", "1", "--python", "3.13",
        "--torch", "2.14", "--commit", "1234567", "--entorno", "colab-t4", "--modelo", "2024",
    )  # fmt: skip

    datos = cabecera(boveda)

    for campo, esperado in {
        "dataset_version": "1", "python": "3.13", "torch": "2.14", "commit": "1234567",
        "entorno": "colab-t4", "modelo": "2024",
    }.items():  # fmt: skip
        assert datos[campo] == esperado and isinstance(datos[campo], str), campo


@pytest.mark.parametrize(
    ("escrito", "esperado"), [("0.35", 0.35), ("2", 2), ("-1.5", -1.5), ("1e-3", 0.001)]
)
def test_valor_se_escribe_como_numero(boveda, escrito, esperado):
    registrar(boveda, "nuevo")

    orden = ("campos", "--exp", "exp001", "--set", f"valor={escrito}")

    codigo = registrar(boveda, *orden)

    valor = cabecera(boveda)["valor"]
    assert codigo == 0 and isinstance(valor, int | float) and not isinstance(valor, bool)
    assert valor == esperado


def test_la_metrica_principal_es_texto_y_estado_tambien(boveda):
    registrar(boveda, "nuevo")

    registrar(boveda, "campos", "--exp", "exp001", "--set", "metrica_principal=mAP50",
              "--set", "estado=hecho")  # fmt: skip

    datos = cabecera(boveda)
    assert datos["metrica_principal"] == "mAP50" and datos["estado"] == "hecho"


@pytest.mark.parametrize("valor", ["0,35", "85%", "abc", "0.35 mAP"])
def test_un_valor_que_no_es_un_numero_se_rechaza_y_la_nota_no_cambia(
    boveda, capsys, valor
):
    registrar(boveda, "nuevo")
    antes = texto(boveda)

    codigo = registrar(boveda, "campos", "--exp", "exp001", "--set", f"valor={valor}")

    assert codigo == 2 and "valor" in capsys.readouterr().err
    assert texto(boveda) == antes


def test_tags_sigue_siendo_una_lista_tras_todas_las_ordenes(
    boveda, escribir_json, tmp_path
):
    registrar(boveda, "nuevo")
    registrar(boveda, *EJECUCION, str(_runs(tmp_path, escribir_json)))
    registrar(boveda, "campos", "--exp", "exp001", "--set", "estado=hecho")

    assert cabecera(boveda)["tags"] == ["experimento"]


def test_las_notas_se_escriben_en_utf8_con_saltos_lf(boveda):
    registrar(boveda, "nuevo", "--objetivo", "evaluación con ñandú")

    crudo = (boveda.carpeta / "exp001.md").read_bytes()

    assert b"\r\n" not in crudo and "ñandú".encode() in crudo
    assert cabecera(boveda)["objetivo"] == "evaluación con ñandú"


# --- marcadores: no rompen la vista de la nota -------------------------------------------


@pytest.fixture
def nota_completa(boveda, escribir_json, tmp_path):
    """Una nota con bloque de configuración y bloque de resultados."""
    config = tmp_path / "v0.yaml"
    config.write_text("detector: whole\n", encoding="utf-8")
    registrar(boveda, "nuevo", "--config", str(config))
    registrar(boveda, *EJECUCION, str(_runs(tmp_path, escribir_json)))
    return boveda


def test_los_marcadores_van_solos_en_su_linea_con_lineas_en_blanco_alrededor(nota_completa):
    lineas = texto(nota_completa).split("\n")
    marcadores = [i for i, linea in enumerate(lineas) if "<!--" in linea or "-->" in linea]

    assert len(marcadores) == 4  # config y resultados, con su inicio y su fin
    for i in marcadores:
        assert lineas[i].startswith("<!-- ") and lineas[i].endswith(" -->")  # nada más en la línea
        assert lineas[i - 1] == "" and lineas[i + 1] == ""


def test_la_tabla_queda_contigua_y_fuera_de_los_marcadores_de_codigo(nota_completa):
    lineas = texto(nota_completa).split("\n")
    inicio, fin = lineas.index(INICIO), lineas.index(FIN)

    tabla = [i for i in range(inicio, fin) if lineas[i].startswith("|")]

    assert len(tabla) == 3  # cabecera, separador y una fila
    assert tabla == list(range(tabla[0], tabla[0] + 3))  # sin huecos entre sus líneas
    dentro_de_codigo, vallas = False, 0
    for linea in lineas:
        if linea.startswith("```"):
            dentro_de_codigo, vallas = not dentro_de_codigo, vallas + 1
        elif dentro_de_codigo:
            assert "<!--" not in linea and "-->" not in linea  # ningún marcador dentro del YAML
    assert vallas == 2 and not dentro_de_codigo


def test_un_parser_de_markdown_reconoce_la_tabla_y_los_comentarios(nota_completa):
    markdown_it = pytest.importorskip("markdown_it")
    parser = markdown_it.MarkdownIt("commonmark").enable("table")

    tokens = parser.parse(texto(nota_completa))

    tipos = [t.type for t in tokens]
    assert tipos.count("table_open") == 1
    filas = [t for t in tokens if t.type == "tr_open"]
    assert len(filas) == 2  # la cabecera y la fila de la ejecución
    comentarios = [t.content.strip() for t in tokens if t.type == "html_block"]
    assert comentarios.count(INICIO) == 1 and comentarios.count(FIN) == 1
    assert [t.info for t in tokens if t.type == "fence"] == ["yaml"]
