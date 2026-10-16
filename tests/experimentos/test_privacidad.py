"""Privacidad: ninguna nota ni salida del script lleva rutas absolutas ni el nombre de usuario."""

import getpass
from datetime import date
from pathlib import Path

import pytest

from src.experimentos.privacidad import (
    contiene_ruta_absoluta,
    contiene_usuario,
    relativa,
    sanear,
    validar,
)
from src.experimentos.registro import main

USUARIO_FICTICIO = "usuario_prueba"
USUARIO_REAL = getpass.getuser()  # el de esta máquina: tampoco debe aparecer nunca


@pytest.fixture(autouse=True)
def usuario_ficticio(monkeypatch):
    monkeypatch.setattr(getpass, "getuser", lambda: USUARIO_FICTICIO)


# --- detección ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "texto",
    [
        r"C:\Users\ana\proyecto\runs",
        "C:/Users/ana/x",
        r"D:\datos\lechugas.csv",
        r"\\servidor\recurso\x",
        "/home/ana/x",
        "/Users/ana/x",
        "/content/drive/MyDrive/x",
        "se guardó en D:/datos/f.csv y ya",
        "ruta='/tmp/x/y'",
        "No such file: 'C:\\\\Users\\\\ana\\\\x'",  # como lo escribe un OSError (barras dobles)
    ],
)
def test_detecta_rutas_absolutas_de_windows_y_de_linux(texto):
    assert contiene_ruta_absoluta(texto)


@pytest.mark.parametrize(
    "texto",
    [
        "runs/v0_owlvit",
        "docs/experimentos/exp001.md",
        "MyDrive/tfg-lechugas/runs/v0",
        "configs/v0.yaml",
        "a/b",
        "1/2/3",
        "https://huggingface.co/google/owlvit-base-patch32",
        "owlvit / whole",
        "mAP@0.5:0.95",
        "color: rojo",
        "",
    ],
)
def test_no_confunde_con_rutas_absolutas_lo_que_son_rutas_relativas_o_texto(texto):
    assert not contiene_ruta_absoluta(texto)


def test_detecta_el_nombre_de_usuario_como_palabra_completa_y_sin_distinguir_mayusculas():
    assert contiene_usuario(f"hecho por {USUARIO_FICTICIO}")
    assert contiene_usuario(f"hecho por {USUARIO_FICTICIO.upper()}")
    assert contiene_usuario(rf"C:\Users\{USUARIO_FICTICIO}\x")
    assert not contiene_usuario(f"{USUARIO_FICTICIO}s es otra palabra")
    assert not contiene_usuario("sin nombres")


def test_un_nombre_de_usuario_muy_corto_se_ignora(monkeypatch):
    monkeypatch.setattr(getpass, "getuser", lambda: "ab")

    assert not contiene_usuario("tabla ab cd")


def test_si_no_se_puede_saber_el_usuario_no_falla(monkeypatch):
    def sin_usuario():
        raise OSError("sin usuario")

    monkeypatch.setattr(getpass, "getuser", sin_usuario)

    assert not contiene_usuario("hola")


# --- validar ------------------------------------------------------------------------------


@pytest.mark.parametrize("valor", [r"C:\Users\ana\x", "/home/ana/x", f"nota de {USUARIO_FICTICIO}"])
def test_validar_rechaza_rutas_absolutas_y_el_usuario_sin_repetirlos_en_el_mensaje(valor):
    with pytest.raises(ValueError, match="objetivo") as error:
        validar("objetivo", valor)

    assert "ana" not in str(error.value) and USUARIO_FICTICIO not in str(error.value)


def test_validar_acepta_texto_normal_y_rutas_relativas():
    for valor in ("validar el pipeline", "configs/v0.yaml", "MyDrive/tfg-lechugas/runs", ""):
        validar("objetivo", valor)  # no lanza


# --- sanear y relativa --------------------------------------------------------------------


def test_sanear_quita_la_raiz_del_repositorio(tmp_path):
    ruta = tmp_path / "docs" / "experimentos" / "exp001.md"

    salida = sanear(f"Creada: {ruta}", tmp_path)

    assert str(tmp_path) not in salida
    assert "exp001.md" in salida and "docs" in salida


def test_sanear_sustituye_otras_rutas_absolutas_y_el_usuario():
    salida = sanear(rf"Error en C:\Otro\lugar\f.txt de {USUARIO_FICTICIO}", None)

    assert "Otro" not in salida and "lugar" not in salida
    assert USUARIO_FICTICIO not in salida and "C:\\" not in salida
    assert "<ruta>" in salida and "<usuario>" in salida


def test_sanear_deja_intacto_lo_que_ya_es_seguro():
    texto = "Registrada en exp001: whole / CPU local, 3 imágenes. docs/experimentos/exp001.md"

    assert sanear(texto, Path("/algun/sitio")) == texto


def test_sanear_es_idempotente(tmp_path):
    una_vez = sanear(f"{tmp_path} y {USUARIO_FICTICIO}", tmp_path)

    assert sanear(una_vez, tmp_path) == una_vez


def test_relativa_da_una_ruta_con_barras_normales(tmp_path):
    assert relativa(tmp_path / "configs" / "v0.yaml", tmp_path) == "configs/v0.yaml"
    assert relativa(tmp_path, tmp_path) == "."


def test_relativa_fuera_de_la_raiz_lanza_value_error(tmp_path):
    with pytest.raises(ValueError, match="dentro del repositorio"):
        relativa(tmp_path.parent / "otro.yaml", tmp_path)


# --- el flujo completo de la línea de comandos --------------------------------------------


def registrar(docs, *args):
    return main(["--docs", str(docs.raiz), *args], hoy=date(2026, 10, 7))


def _runs(tmp_path, escribir_json, **opciones):
    for i in range(2):
        escribir_json(tmp_path / "runs", f"f{i}", total_ms=10 * (i + 1), **opciones)
    return tmp_path / "runs"


def _sin_rastro(texto, tmp_path):
    """True si el texto no contiene el directorio temporal ni ningún nombre de usuario."""
    minusculas = texto.lower()
    return not any(
        valor in minusculas
        for valor in (
            str(tmp_path).lower(),
            tmp_path.as_posix().lower(),
            USUARIO_FICTICIO,
            USUARIO_REAL.lower(),
        )
    )


def test_el_flujo_completo_no_deja_rutas_absolutas_ni_usuario_en_notas_ni_salidas(
    docs, escribir_json, tmp_path, capsys
):
    config = tmp_path / "v0.yaml"
    config.write_text("detector: whole\n", encoding="utf-8")
    runs = _runs(tmp_path, escribir_json, completo=True, env={"dirty": True})

    codigos = [
        registrar(docs, "nuevo", "--config", str(config), "--objetivo", "validar"),
        registrar(docs, "ejecucion", "--exp", "exp001", "--runs", str(runs)),
        registrar(docs, "campos", "--exp", "exp001", "--set", "estado=hecho"),
    ]

    salida = capsys.readouterr()
    nota = (docs.carpeta / "exp001.md").read_text(encoding="utf-8")
    assert codigos == [0, 0, 0]
    assert "sin commitear" in salida.err  # con dirty=true hay avisos, y tampoco llevan rutas
    assert _sin_rastro(salida.out, tmp_path), salida.out
    assert _sin_rastro(salida.err, tmp_path), salida.err
    assert _sin_rastro(nota, tmp_path), nota


def test_la_ruta_que_se_muestra_al_crear_es_relativa(docs, capsys):
    registrar(docs, "nuevo")

    salida = capsys.readouterr().out

    assert "Creada: docs/experimentos/exp001.md" in salida.replace("\\", "/")


def test_config_con_ruta_absoluta_dentro_del_repositorio_se_guarda_relativa(
    docs, tmp_path
):
    (tmp_path / "configs").mkdir()
    config = tmp_path / "configs" / "v0.yaml"
    config.write_text("detector: whole\n", encoding="utf-8")

    codigo = registrar(docs, "nuevo", "--config", str(config))

    nota = (docs.carpeta / "exp001.md").read_text(encoding="utf-8")
    assert codigo == 0 and "config: configs/v0.yaml" in nota.split("\n")
    assert _sin_rastro(nota, tmp_path)


def test_config_fuera_del_repositorio_se_rechaza_y_no_se_crea_la_nota(
    docs, tmp_path_factory, capsys
):
    fuera = tmp_path_factory.mktemp("fuera") / "v0.yaml"
    fuera.write_text("detector: whole\n", encoding="utf-8")

    codigo = registrar(docs, "nuevo", "--config", str(fuera))

    error = capsys.readouterr().err
    assert codigo == 2 and "dentro del repositorio" in error
    assert _sin_rastro(error, fuera.parent)
    assert list(docs.carpeta.iterdir()) == []


def test_un_yaml_con_una_ruta_absoluta_no_se_copia_a_la_nota(docs, tmp_path, capsys):
    config = tmp_path / "v0.yaml"
    config.write_text("datos: C:/Users/ana/lechugas\n", encoding="utf-8")

    codigo = registrar(docs, "nuevo", "--config", str(config))

    assert codigo == 2 and "ruta absoluta" in capsys.readouterr().err
    assert list(docs.carpeta.iterdir()) == []


@pytest.mark.parametrize(
    "valor", [r"C:\Users\ana\x", "/home/ana/x", f"hecho por {USUARIO_FICTICIO}"]
)
def test_un_campo_con_ruta_absoluta_o_usuario_se_rechaza_y_la_nota_no_cambia(
    docs, capsys, valor
):
    registrar(docs, "nuevo")
    antes = (docs.carpeta / "exp001.md").read_text(encoding="utf-8")
    capsys.readouterr()

    codigo = registrar(docs, "campos", "--exp", "exp001", "--set", f"objetivo={valor}")

    error = capsys.readouterr().err
    assert codigo == 2 and "objetivo" in error
    assert "ana" not in error and USUARIO_FICTICIO not in error
    assert (docs.carpeta / "exp001.md").read_text(encoding="utf-8") == antes


@pytest.mark.parametrize("opcion", ["--observacion", "--commit"])
@pytest.mark.parametrize("valor", [r"C:\Users\ana\runs", f"de {USUARIO_FICTICIO}"])
def test_las_opciones_de_ejecucion_tambien_se_validan(
    docs, escribir_json, tmp_path, capsys, opcion, valor
):
    registrar(docs, "nuevo")
    antes = (docs.carpeta / "exp001.md").read_text(encoding="utf-8")
    runs = _runs(tmp_path, escribir_json, completo=True)

    codigo = registrar(docs, "ejecucion", "--exp", "exp001", "--runs", str(runs), opcion, valor)

    error = capsys.readouterr().err
    assert codigo == 2 and ("ruta absoluta" in error or "usuario" in error)
    assert "ana" not in error and USUARIO_FICTICIO not in error
    assert (docs.carpeta / "exp001.md").read_text(encoding="utf-8") == antes


def test_un_dato_del_json_con_una_ruta_absoluta_no_llega_a_la_nota(
    docs, escribir_json, tmp_path, capsys
):
    registrar(docs, "nuevo")
    antes = (docs.carpeta / "exp001.md").read_text(encoding="utf-8")
    env = {"detector": r"C:\Users\ana\modelo.pt", "device": "cpu"}
    runs = _runs(tmp_path, escribir_json, env=env)

    codigo = registrar(docs, "ejecucion", "--exp", "exp001", "--runs", str(runs))

    assert codigo == 2 and "ruta absoluta" in capsys.readouterr().err
    assert (docs.carpeta / "exp001.md").read_text(encoding="utf-8") == antes


def test_los_errores_del_sistema_operativo_no_cuelan_rutas_absolutas(
    docs, tmp_path, capsys
):
    inexistente = tmp_path / USUARIO_FICTICIO / "no_existe"

    codigo = registrar(docs, "nuevo", "--config", str(inexistente / "v0.yaml"))
    error_config = capsys.readouterr().err
    sin_plantilla = tmp_path / "sin_plantilla"
    codigo_2 = main(["--docs", str(sin_plantilla), "nuevo"], hoy=date(2026, 10, 7))
    error_plantilla = capsys.readouterr().err

    assert (codigo, codigo_2) == (2, 2)
    assert _sin_rastro(error_config, tmp_path), error_config
    assert _sin_rastro(error_plantilla, tmp_path), error_plantilla


def test_runs_inexistente_no_cuela_su_ruta_absoluta(docs, tmp_path, capsys):
    registrar(docs, "nuevo")
    capsys.readouterr()
    runs = tmp_path / USUARIO_FICTICIO / "no_hay"

    codigo = registrar(docs, "ejecucion", "--exp", "exp001", "--runs", str(runs))

    error = capsys.readouterr().err
    assert codigo == 2
    assert _sin_rastro(error, tmp_path), error


def test_runs_fuera_del_repositorio_se_rechaza_sin_rastro_de_su_ruta(
    docs, escribir_json, tmp_path_factory, capsys
):
    registrar(docs, "nuevo")
    antes = (docs.carpeta / "exp001.md").read_text(encoding="utf-8")
    fuera = tmp_path_factory.mktemp("fuera") / "runs"
    escribir_json(fuera, "a", completo=True)
    capsys.readouterr()

    codigo = registrar(docs, "ejecucion", "--exp", "exp001", "--runs", str(fuera))

    error = capsys.readouterr().err
    assert codigo == 2 and "dentro del repositorio" in error
    assert _sin_rastro(error, fuera.parent)
    assert (docs.carpeta / "exp001.md").read_text(encoding="utf-8") == antes


def test_la_carpeta_de_runs_se_guarda_relativa_y_la_nota_no_lleva_rastro(
    docs, escribir_json, tmp_path
):
    registrar(docs, "nuevo")
    runs = _runs(tmp_path, escribir_json, completo=True)  # ruta absoluta en la línea de comandos

    registrar(docs, "ejecucion", "--exp", "exp001", "--runs", str(runs))

    nota = (docs.carpeta / "exp001.md").read_text(encoding="utf-8")
    assert "| runs |" in nota  # la carpeta, relativa a la raíz del repo
    assert _sin_rastro(nota, tmp_path), nota
