"""Pruebas de src/experimentos/nota.py: crear notas y editarlas sin tocar lo demás."""

from datetime import date

import pytest

from src.experimentos import nota
from src.experimentos.nota import (
    campos_de_plantilla,
    crear_nota,
    leer_bloque,
    leer_cabecera,
    modificar_nota,
    poner_bloque,
    poner_campos,
    siguiente_id,
)

CAMPOS = (
    "id", "fecha", "estado", "objetivo", "dataset", "dataset_version", "modelo", "config",
    "commit", "entorno", "python", "torch", "metrica_principal", "valor", "carpeta_drive", "tags",
)  # fmt: skip
EDITABLES = set(CAMPOS) - {"id", "tags"}

NOTA = """---
id: exp001
fecha: 2026-10-07
estado: planificado
objetivo:
dataset:
carpeta_drive:
valor:
tags:
  - experimento
---
Texto suelto antes del primer título.
# exp001

## Objetivo e hipótesis
Verificar que funciona.
## Datos

## Resultados
Texto que escribió el autor.
## Análisis de errores

## Conclusión y siguiente paso
Siguiente paso: algo.
"""


# --- siguiente_id -----------------------------------------------------------------------


def test_siguiente_id_en_carpeta_vacia_es_exp001(tmp_path):
    assert siguiente_id(tmp_path) == "exp001"


def test_siguiente_id_sigue_al_mayor_e_ignora_lo_ajeno(tmp_path):
    for nombre in ("exp000.md", "exp001.md", "exp003.md", "notas.md", "exp1.md", "exp002.txt"):
        (tmp_path / nombre).write_text("", encoding="utf-8")

    assert siguiente_id(tmp_path) == "exp004"


def test_siguiente_id_pasa_de_999(tmp_path):
    (tmp_path / "exp999.md").write_text("", encoding="utf-8")

    assert siguiente_id(tmp_path) == "exp1000"


# --- campos_de_plantilla y leer_cabecera ------------------------------------------------


def test_campos_de_plantilla_en_su_orden(docs):
    assert campos_de_plantilla(docs.plantilla.read_text(encoding="utf-8")) == CAMPOS


def test_leer_cabecera_devuelve_los_valores_tipados():
    cabecera = leer_cabecera(NOTA)

    assert cabecera["id"] == "exp001"
    assert cabecera["fecha"] == date(2026, 10, 7)
    assert cabecera["objetivo"] is None  # vacío: sin dato
    assert cabecera["tags"] == ["experimento"]


def test_leer_cabecera_sin_cabecera_lanza_value_error():
    with pytest.raises(ValueError, match="cabecera"):
        leer_cabecera("# Solo un título\n")


# --- crear_nota -------------------------------------------------------------------------


def test_crear_nota_sustituye_los_marcadores_y_pone_los_campos(docs):
    campos = {"objetivo": "validar el pipeline", "dataset": "fotos propias"}

    ruta = crear_nota(docs.carpeta, docs.plantilla, campos, date(2026, 10, 7))

    texto = ruta.read_text(encoding="utf-8")
    cabecera = leer_cabecera(texto)
    assert ruta == docs.carpeta / "exp001.md"
    assert "{{" not in texto and "# exp001" in texto
    assert cabecera["id"] == "exp001" and cabecera["fecha"] == date(2026, 10, 7)
    assert cabecera["estado"] == "planificado"
    assert cabecera["objetivo"] == "validar el pipeline" and cabecera["dataset"] == "fotos propias"


def test_crear_nota_deja_vacio_lo_que_no_se_da(docs):
    ruta = crear_nota(docs.carpeta, docs.plantilla, {}, date(2026, 10, 7))

    cabecera = leer_cabecera(ruta.read_text(encoding="utf-8"))
    vacios = [c for c in CAMPOS if cabecera[c] is None]
    assert set(vacios) == {
        "objetivo", "dataset", "dataset_version", "modelo", "config", "commit", "entorno",
        "python", "torch", "metrica_principal", "valor", "carpeta_drive",
    }  # fmt: skip


def test_crear_nota_numera_la_siguiente(docs):
    primera = crear_nota(docs.carpeta, docs.plantilla, {}, date(2026, 10, 7))
    segunda = crear_nota(docs.carpeta, docs.plantilla, {}, date(2026, 10, 8))

    assert (primera.name, segunda.name) == ("exp001.md", "exp002.md")


def test_crear_nota_nunca_sobrescribe(docs, monkeypatch):
    existente = docs.carpeta / "exp001.md"
    existente.write_text("contenido del autor", encoding="utf-8")
    monkeypatch.setattr(nota, "siguiente_id", lambda carpeta: "exp001")  # forzar el choque

    with pytest.raises(FileExistsError):
        crear_nota(docs.carpeta, docs.plantilla, {}, date(2026, 10, 7))

    assert existente.read_text(encoding="utf-8") == "contenido del autor"


def test_crear_nota_con_un_marcador_desconocido_lanza_value_error(docs):
    docs.plantilla.write_text("---\nid: x\n---\n{{otro}}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="otro"):
        crear_nota(docs.carpeta, docs.plantilla, {}, date(2026, 10, 7))


def test_crear_nota_rechaza_campos_que_no_estan_en_la_plantilla(docs):
    with pytest.raises(ValueError, match="velocidad"):
        crear_nota(docs.carpeta, docs.plantilla, {"velocidad": "alta"}, date(2026, 10, 7))

    assert list(docs.carpeta.iterdir()) == []


def test_crear_nota_escribe_utf8_con_saltos_lf(docs):
    ruta = crear_nota(docs.carpeta, docs.plantilla, {"objetivo": "evaluación"}, date(2026, 10, 7))

    crudo = ruta.read_bytes()
    assert "evaluación".encode("utf-8") in crudo and b"\r\n" not in crudo


# --- poner_campos -----------------------------------------------------------------------


def test_poner_campos_cambia_solo_las_lineas_pedidas():
    nuevo = poner_campos(NOTA, {"estado": "hecho", "valor": "0.35"}, EDITABLES)

    antes, despues = NOTA.splitlines(), nuevo.splitlines()
    assert len(antes) == len(despues)
    cambiadas = [(a, b) for a, b in zip(antes, despues, strict=True) if a != b]
    assert cambiadas == [("estado: planificado", "estado: hecho"), ("valor:", "valor: 0.35")]


def test_poner_campos_anade_una_clave_que_falta_dentro_de_la_cabecera():
    sin_drive = NOTA.replace("carpeta_drive:\n", "")

    nuevo = poner_campos(sin_drive, {"carpeta_drive": "MyDrive/tfg-lechugas/runs"}, EDITABLES)

    cabecera = leer_cabecera(nuevo)
    assert cabecera["carpeta_drive"] == "MyDrive/tfg-lechugas/runs"
    assert nuevo.endswith(sin_drive.split("---\n", 2)[2])  # el cuerpo queda igual


def test_poner_campos_con_clave_desconocida_lanza_value_error_que_la_nombra():
    with pytest.raises(ValueError, match="velocidad") as error:
        poner_campos(NOTA, {"velocidad": "alta"}, EDITABLES)

    assert "estado" in str(error.value)  # y lista las válidas


@pytest.mark.parametrize("clave", ["id", "tags"])
def test_poner_campos_no_edita_id_ni_tags(clave):
    with pytest.raises(ValueError, match=clave):
        poner_campos(NOTA, {clave: "x"}, EDITABLES)


def test_poner_campos_con_un_salto_de_linea_lanza_value_error():
    with pytest.raises(ValueError, match="objetivo"):
        poner_campos(NOTA, {"objetivo": "línea 1\nlínea 2"}, EDITABLES)


@pytest.mark.parametrize(
    ("valor", "linea"),
    [
        ("evaluación del corte", "objetivo: evaluación del corte"),
        ("colab-t4", "objetivo: colab-t4"),
        ("2.11.0+cu130", "objetivo: 2.11.0+cu130"),
        ("3.12.10", "objetivo: 3.12.10"),
        ("abc1234", "objetivo: abc1234"),
        # Todo lo que YAML leería como número, fecha, booleano... se entrecomilla: un campo de
        # texto (un commit, una versión) tiene que volver a leerse como texto.
        ("1234567", 'objetivo: "1234567"'),
        ("0123456", 'objetivo: "0123456"'),
        ("0x1f", 'objetivo: "0x1f"'),
        ("0.35", 'objetivo: "0.35"'),
        ("3.13", 'objetivo: "3.13"'),
        ("2026-10-07", 'objetivo: "2026-10-07"'),
        ("true", 'objetivo: "true"'),
        ("null", 'objetivo: "null"'),
        ("a: b", 'objetivo: "a: b"'),
        ("a #b", 'objetivo: "a #b"'),
        ("- uno", 'objetivo: "- uno"'),
        ("[1, 2]", 'objetivo: "[1, 2]"'),
    ],
)
def test_los_campos_de_texto_se_escriben_como_texto(valor, linea):
    nuevo = poner_campos(NOTA, {"objetivo": valor}, EDITABLES)

    assert linea in nuevo.splitlines()
    leido = leer_cabecera(nuevo)["objetivo"]
    assert isinstance(leido, str) and leido == valor  # vuelve a leerse igual, como texto


def test_un_campo_de_texto_vacio_queda_sin_dato():
    nuevo = poner_campos(NOTA, {"objetivo": ""}, EDITABLES)

    assert "objetivo:" in nuevo.splitlines()
    assert leer_cabecera(nuevo)["objetivo"] is None


@pytest.mark.parametrize(
    ("valor", "linea", "esperado"),
    [
        ("0.35", "valor: 0.35", 0.35),
        ("-2", "valor: -2", -2),
        ("2", "valor: 2", 2),
        (".5", "valor: 0.5", 0.5),
        ("5.", "valor: 5", 5),
        ("1e-3", "valor: 0.001", 0.001),  # YAML leería "1e-3" como texto: se escribe 0.001
        ("1e3", "valor: 1000", 1000),
    ],
)
def test_valor_se_escribe_como_numero(valor, linea, esperado):
    nuevo = poner_campos(NOTA, {"valor": valor}, EDITABLES)

    assert linea in nuevo.splitlines()
    leido = leer_cabecera(nuevo)["valor"]
    assert isinstance(leido, int | float) and not isinstance(leido, bool)
    assert leido == esperado


@pytest.mark.parametrize(
    "valor", ["0,35", "85%", "abc", "nan", "inf", "1.2.3", "0x10", "--5", "1 2", "0.35 f1"]
)
def test_valor_que_no_es_un_numero_da_error_y_no_toca_la_nota(valor):
    with pytest.raises(ValueError, match="valor"):
        poner_campos(NOTA, {"valor": valor, "estado": "hecho"}, EDITABLES)


def test_valor_vacio_es_sin_dato():
    nuevo = poner_campos(NOTA, {"valor": ""}, EDITABLES)

    assert leer_cabecera(nuevo)["valor"] is None


def test_fecha_se_escribe_como_fecha_iso_y_se_valida():
    nuevo = poner_campos(NOTA, {"fecha": "2026-11-01"}, EDITABLES)

    assert "fecha: 2026-11-01" in nuevo.splitlines()
    assert leer_cabecera(nuevo)["fecha"] == date(2026, 11, 1)
    with pytest.raises(ValueError, match="fecha"):
        poner_campos(NOTA, {"fecha": "01/11/2026"}, EDITABLES)


def test_modificar_nota_conserva_los_saltos_de_linea_crlf(tmp_path):
    ruta = tmp_path / "exp001.md"
    ruta.write_bytes(b"---\r\nestado: planificado\r\n---\r\n# x\r\n")

    modificar_nota(ruta, lambda texto: poner_campos(texto, {"estado": "hecho"}, {"estado"}))

    assert ruta.read_bytes() == b"---\r\nestado: hecho\r\n---\r\n# x\r\n"


def test_modificar_nota_conserva_los_saltos_de_linea_lf(tmp_path):
    ruta = tmp_path / "exp001.md"
    ruta.write_bytes(b"---\nestado: planificado\n---\n# x\n")

    modificar_nota(ruta, lambda texto: poner_campos(texto, {"estado": "hecho"}, {"estado"}))

    assert ruta.read_bytes() == b"---\nestado: hecho\n---\n# x\n"


# --- bloques automáticos ----------------------------------------------------------------

INICIO = "<!-- registro:resultados -->"
FIN = "<!-- /registro:resultados -->"


def test_poner_bloque_se_inserta_justo_tras_el_encabezado_de_su_seccion():
    nuevo = poner_bloque(NOTA, "resultados", "contenido automático", "Resultados")

    lineas = nuevo.splitlines()
    i = lineas.index("## Resultados")
    assert lineas[i + 1 : i + 8] == ["", INICIO, "", "contenido automático", "", FIN, ""]
    assert lineas[i + 8] == "Texto que escribió el autor."


def test_poner_bloque_no_toca_nada_fuera_del_bloque():
    nuevo = poner_bloque(NOTA, "resultados", "x", "Resultados")

    # Quitando el bloque y sus líneas en blanco de separación, queda la nota original.
    sin_bloque = nuevo.replace(f"\n{INICIO}\n\nx\n\n{FIN}\n\n", "")
    assert sin_bloque == NOTA


def test_los_marcadores_van_solos_en_su_linea_con_una_linea_en_blanco_alrededor():
    # Así un renderizador de Markdown no los pega a la tabla ni al texto de al lado.
    tabla = "| a | b |\n| - | - |\n| 1 | 2 |"

    lineas = poner_bloque(NOTA, "resultados", tabla, "Resultados").splitlines()

    i, f = lineas.index(INICIO), lineas.index(FIN)
    assert lineas[i - 1] == "" and lineas[i + 1] == ""
    assert lineas[f - 1] == "" and lineas[f + 1] == ""
    assert lineas[i + 2 : f - 1] == ["| a | b |", "| - | - |", "| 1 | 2 |"]  # la tabla, contigua


def test_poner_bloque_repetido_reemplaza_y_no_duplica():
    una_vez = poner_bloque(NOTA, "resultados", "viejo", "Resultados")

    dos_veces = poner_bloque(una_vez, "resultados", "nuevo", "Resultados")

    assert dos_veces.count(INICIO) == 1 and dos_veces.count(FIN) == 1
    assert "viejo" not in dos_veces and "nuevo" in dos_veces
    assert "Texto que escribió el autor." in dos_veces


def test_poner_bloque_repetido_con_lo_mismo_no_cambia_nada():
    una_vez = poner_bloque(NOTA, "resultados", "igual", "Resultados")

    assert poner_bloque(una_vez, "resultados", "igual", "Resultados") == una_vez


def test_el_texto_del_autor_dentro_de_la_seccion_se_conserva_al_reemplazar():
    con_bloque = poner_bloque(NOTA, "resultados", "viejo", "Resultados")
    editada = con_bloque.replace("Texto que escribió el autor.", "Texto cambiado a mano.")

    nuevo = poner_bloque(editada, "resultados", "nuevo", "Resultados")

    assert "Texto cambiado a mano." in nuevo


def test_bloques_de_secciones_distintas_son_independientes():
    con_resultados = poner_bloque(NOTA, "resultados", "R", "Resultados")

    con_ambos = poner_bloque(con_resultados, "config", "C", "Datos")

    assert leer_bloque(con_ambos, "resultados") == "R"
    assert leer_bloque(con_ambos, "config") == "C"
    assert con_ambos.index("<!-- registro:config -->") < con_ambos.index(INICIO)


def test_leer_bloque_inexistente_devuelve_none():
    assert leer_bloque(NOTA, "resultados") is None


def test_poner_bloque_sin_la_seccion_lanza_value_error_que_la_nombra():
    with pytest.raises(ValueError, match="Inexistente"):
        poner_bloque(NOTA, "resultados", "x", "Inexistente")


def test_poner_bloque_con_marcador_de_inicio_sin_fin_lanza_value_error():
    roto = NOTA.replace("## Resultados\n", f"## Resultados\n{INICIO}\n")

    with pytest.raises(ValueError, match="registro:resultados"):
        poner_bloque(roto, "resultados", "x", "Resultados")


def test_el_contenido_se_guarda_sin_saltos_de_linea_sobrantes():
    nuevo = poner_bloque(NOTA, "resultados", "\n\nlínea\n\n", "Resultados")

    assert leer_bloque(nuevo, "resultados") == "línea"
