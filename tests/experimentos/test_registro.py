"""Pruebas de la línea de comandos: python -m src.experimentos.registro (main(argv))."""

import inspect
from datetime import date
from pathlib import Path

import pytest

from src.experimentos.ejecuciones import parsear_tabla
from src.experimentos.nota import leer_bloque, leer_cabecera
from src.experimentos.registro import crear_parser, main

HOY = date(2026, 10, 7)
RAIZ = Path(__file__).resolve().parents[2]


def ejecutar(docs, *args):
    """Lanza main con la bóveda de prueba y la fecha fija."""
    return main(["--docs", str(docs.raiz), *args], hoy=HOY)


def _cabecera(docs, exp="exp001"):
    return leer_cabecera((docs.carpeta / f"{exp}.md").read_text(encoding="utf-8"))


def _texto(docs, exp="exp001"):
    return (docs.carpeta / f"{exp}.md").read_text(encoding="utf-8")


def _filas(docs, exp="exp001"):
    return parsear_tabla(leer_bloque(_texto(docs, exp), "resultados") or "")


def _runs(tmp_path, escribir_json, nombre="runs", n=3, **opciones):
    """Una carpeta de JSON completos (como los del pipeline de hoy) dentro de la raíz de pruebas."""
    opciones.setdefault("completo", True)
    for i in range(n):
        escribir_json(
            tmp_path / nombre, f"f{i}", total_ms=10 * (i + 1), detect_ms=5 * (i + 1), **opciones
        )
    return tmp_path / nombre


def _ejecucion(docs, runs, *extra):
    return ejecutar(docs, "ejecucion", "--exp", "exp001", "--runs", str(runs), *extra)


# --- nuevo ------------------------------------------------------------------------------


def test_nuevo_crea_la_nota_y_dice_que_falta(docs, capsys):
    codigo = ejecutar(docs, "nuevo", "--objetivo", "validar la v0")

    salida = capsys.readouterr().out
    assert codigo == 0 and (docs.carpeta / "exp001.md").is_file()
    assert "exp001" in salida
    faltan = next(linea for linea in salida.splitlines() if linea.startswith("Faltan"))
    assert "dataset" in faltan and "modelo" in faltan
    assert "objetivo" not in faltan  # este sí se dio
    assert _cabecera(docs)["objetivo"] == "validar la v0"


def test_nuevo_no_inventa_nada_de_lo_que_no_se_da(docs):
    ejecutar(docs, "nuevo")

    cabecera = _cabecera(docs)
    for campo in ("commit", "entorno", "python", "torch", "valor", "metrica_principal"):
        assert cabecera[campo] is None, campo
    assert cabecera["estado"] == "planificado" and cabecera["fecha"] == HOY


def test_nuevo_con_todos_los_campos(docs):
    ejecutar(
        docs, "nuevo", "--objetivo", "o", "--dataset", "d", "--dataset-version", "v1",
        "--modelo", "m", "--entorno", "colab-t4", "--python", "3.13.15", "--torch", "2.11.0+cu130",
        "--carpeta-drive", "MyDrive/tfg-lechugas/runs/v0", "--commit", "def5678",
    )  # fmt: skip

    cabecera = _cabecera(docs)
    assert cabecera["dataset_version"] == "v1" and cabecera["entorno"] == "colab-t4"
    assert cabecera["python"] == "3.13.15" and cabecera["torch"] == "2.11.0+cu130"
    assert cabecera["carpeta_drive"] == "MyDrive/tfg-lechugas/runs/v0"
    assert cabecera["commit"] == "def5678"


def test_nuevo_con_config_guarda_la_ruta_y_una_copia_literal_del_yaml(docs, tmp_path):
    yaml_ejemplo = tmp_path / "v0.yaml"
    yaml_ejemplo.write_text("detector: whole\nowlvit:\n  umbral: 0.1  # ñ\n", encoding="utf-8")

    codigo = ejecutar(docs, "nuevo", "--config", str(yaml_ejemplo))

    assert codigo == 0
    assert _cabecera(docs)["config"] == "v0.yaml"  # relativa a la raíz, nunca absoluta
    copia = leer_bloque(_texto(docs), "config")
    assert copia.startswith("```yaml") and copia.endswith("```")
    assert "detector: whole\nowlvit:\n  umbral: 0.1  # ñ" in copia
    assert _texto(docs).index("## Configuración") < _texto(docs).index("registro:config")


def test_nuevo_con_config_inexistente_falla_y_no_crea_la_nota(docs, capsys):
    codigo = ejecutar(docs, "nuevo", "--config", "no_existe.yaml")

    assert codigo == 2 and "no_existe.yaml" in capsys.readouterr().err
    assert list(docs.carpeta.iterdir()) == []


def test_nuevo_numera_las_notas_seguidas(docs):
    ejecutar(docs, "nuevo")
    ejecutar(docs, "nuevo")

    assert sorted(p.name for p in docs.carpeta.iterdir()) == ["exp001.md", "exp002.md"]


# --- campos -----------------------------------------------------------------------------


def test_campos_actualiza_solo_lo_pedido(docs):
    ejecutar(docs, "nuevo", "--objetivo", "o")
    antes = _cabecera(docs)

    codigo = ejecutar(docs, "campos", "--exp", "exp001", "--set", "estado=hecho",
                      "--set", "valor=0.35")  # fmt: skip

    despues = _cabecera(docs)
    assert codigo == 0
    assert despues["estado"] == "hecho" and despues["valor"] == 0.35
    assert {k for k in antes if antes[k] != despues[k]} == {"estado", "valor"}


def test_campos_con_clave_desconocida_falla_y_la_nombra(docs, capsys):
    ejecutar(docs, "nuevo")

    codigo = ejecutar(docs, "campos", "--exp", "exp001", "--set", "velocidad=alta")

    assert codigo == 2 and "velocidad" in capsys.readouterr().err


def test_campos_sin_igual_falla(docs, capsys):
    ejecutar(docs, "nuevo")

    codigo = ejecutar(docs, "campos", "--exp", "exp001", "--set", "estado")

    assert codigo == 2 and "clave=valor" in capsys.readouterr().err


def test_campos_sobre_una_nota_inexistente_falla(docs, capsys):
    codigo = ejecutar(docs, "campos", "--exp", "exp099", "--set", "estado=hecho")

    assert codigo == 2 and "exp099" in capsys.readouterr().err


def test_un_estado_desconocido_se_acepta_pero_se_avisa(docs, capsys):
    ejecutar(docs, "nuevo")

    codigo = ejecutar(docs, "campos", "--exp", "exp001", "--set", "estado=raro")

    assert codigo == 0 and "estado" in capsys.readouterr().err
    assert _cabecera(docs)["estado"] == "raro"


def test_campos_conserva_el_texto_que_escribio_el_autor(docs):
    ejecutar(docs, "nuevo")
    ruta = docs.carpeta / "exp001.md"
    ruta.write_text(
        ruta.read_text(encoding="utf-8").replace(
            "## Conclusión y siguiente paso", "## Conclusión y siguiente paso\nMi conclusión ñ."
        ),
        encoding="utf-8",
    )

    ejecutar(docs, "campos", "--exp", "exp001", "--set", "estado=hecho")

    assert "Mi conclusión ñ." in _texto(docs)


# --- ejecucion --------------------------------------------------------------------------


def test_ejecucion_anade_la_fila_calculada_desde_los_json(docs, escribir_json, tmp_path):
    ejecutar(docs, "nuevo")
    runs = _runs(tmp_path, escribir_json)

    codigo = _ejecucion(docs, runs)

    fila = _filas(docs)[0]
    assert codigo == 0 and len(_filas(docs)) == 1
    assert fila["Carpeta"] == "runs"  # relativa a la raíz de la bóveda de pruebas
    assert (fila["Detector"], fila["Dispositivo"]) == ("whole", "cpu")  # los lee del JSON
    assert (fila["Commit"], fila["Sucio"]) == ("abc1234", "no")
    assert fila["Imágenes"] == "3" and fila["Tamaño imagen"] == "640x480"
    assert (fila["Repeticiones"], fila["Calentamiento"]) == ("5", "1")
    assert fila["Mediana (ms)"] == "20.00" and fila["Mediana detección (ms)"] == "10.00"
    assert fila["Memoria (MB)"] == "80.0" and fila["Carga modelo (ms)"] == ""  # whole: sin modelo


def test_ejecucion_toma_el_commit_de_la_cabecera_del_json_y_no_toca_nada_mas(
    docs, escribir_json, tmp_path
):
    ejecutar(docs, "nuevo", "--objetivo", "o")
    antes = _cabecera(docs)

    _ejecucion(docs, _runs(tmp_path, escribir_json))

    despues = _cabecera(docs)
    assert despues["commit"] == "abc1234"
    assert {k for k in antes if antes[k] != despues[k]} == {"commit"}  # estado, valor... igual


def test_ejecucion_repetida_no_duplica_y_deja_la_nota_igual(docs, escribir_json, tmp_path):
    ejecutar(docs, "nuevo")
    runs = _runs(tmp_path, escribir_json)

    _ejecucion(docs, runs)
    primera = _texto(docs)
    _ejecucion(docs, runs)

    assert _texto(docs) == primera and len(_filas(docs)) == 1


def test_otra_carpeta_es_otra_fila_aunque_el_detector_y_el_dispositivo_coincidan(
    docs, escribir_json, tmp_path
):
    ejecutar(docs, "nuevo")
    for nombre in ("runs_a", "runs_b", "runs_c"):
        _ejecucion(docs, _runs(tmp_path, escribir_json, nombre))

    filas = _filas(docs)
    assert [f["Carpeta"] for f in filas] == ["runs_a", "runs_b", "runs_c"]
    assert {(f["Detector"], f["Dispositivo"]) for f in filas} == {("whole", "cpu")}


def test_la_carpeta_de_runs_puede_darse_relativa_a_donde_se_lanza(
    docs, escribir_json, tmp_path, monkeypatch
):
    ejecutar(docs, "nuevo")
    _runs(tmp_path, escribir_json, "runs/v0_whole")
    monkeypatch.chdir(tmp_path)

    codigo = ejecutar(docs, "ejecucion", "--exp", "exp001", "--runs", "runs/v0_whole")

    assert codigo == 0 and _filas(docs)[0]["Carpeta"] == "runs/v0_whole"


def test_la_observacion_se_conserva_y_se_puede_cambiar(docs, escribir_json, tmp_path):
    ejecutar(docs, "nuevo")
    runs = _runs(tmp_path, escribir_json)

    _ejecucion(docs, runs, "--observacion", "falla con lechugas pequeñas")
    _ejecucion(docs, runs)
    assert _filas(docs)[0]["Observación"] == "falla con lechugas pequeñas"

    _ejecucion(docs, runs, "--observacion", "otra")
    assert _filas(docs)[0]["Observación"] == "otra"


def test_ya_no_existen_las_opciones_detector_ni_dispositivo(docs, escribir_json, tmp_path):
    ejecutar(docs, "nuevo")
    runs = _runs(tmp_path, escribir_json)

    for opcion, valor in (("--detector", "whole"), ("--dispositivo", "CPU local")):
        with pytest.raises(SystemExit) as salida:
            _ejecucion(docs, runs, opcion, valor)
        assert salida.value.code == 2  # argparse: opción desconocida
    assert _filas(docs) == []


def test_main_ya_no_recibe_el_ejecutor_de_git():
    assert "ejecutar" not in inspect.signature(main).parameters


def test_el_commit_del_json_no_pisa_uno_ya_anotado_pero_lo_comenta(
    docs, escribir_json, tmp_path, capsys
):
    ejecutar(docs, "nuevo", "--commit", "zzz9999")

    _ejecucion(docs, _runs(tmp_path, escribir_json))

    salida = capsys.readouterr()
    assert _cabecera(docs)["commit"] == "zzz9999"
    assert "zzz9999" in salida.out + salida.err and "abc1234" in salida.out + salida.err


def test_commit_explicito_manda_sobre_el_del_json_y_sobre_el_anotado(
    docs, escribir_json, tmp_path
):
    ejecutar(docs, "nuevo", "--commit", "zzz9999")

    _ejecucion(docs, _runs(tmp_path, escribir_json), "--commit", "def5678")

    assert _cabecera(docs)["commit"] == "def5678"


def test_ejecucion_con_carpeta_sin_json_falla_y_no_toca_la_nota(docs, tmp_path, capsys):
    ejecutar(docs, "nuevo")
    antes = _texto(docs)
    vacia = tmp_path / "vacia"
    vacia.mkdir()

    codigo = _ejecucion(docs, vacia)

    assert codigo == 2 and "JSON" in capsys.readouterr().err
    assert _texto(docs) == antes


def test_ejecucion_sobre_una_nota_inexistente_falla(docs, escribir_json, tmp_path, capsys):
    runs = _runs(tmp_path, escribir_json)

    codigo = ejecutar(docs, "ejecucion", "--exp", "exp042", "--runs", str(runs))

    assert codigo == 2 and "exp042" in capsys.readouterr().err


def test_ejecucion_sobre_una_nota_sin_la_seccion_resultados_falla_y_no_la_toca(
    docs, escribir_json, tmp_path, capsys
):
    ejecutar(docs, "nuevo")
    ruta = docs.carpeta / "exp001.md"
    ruta.write_text(
        ruta.read_text(encoding="utf-8").replace("## Resultados", "## Otra cosa"), encoding="utf-8"
    )
    antes = _texto(docs)

    codigo = _ejecucion(docs, _runs(tmp_path, escribir_json))

    assert codigo == 2 and "Resultados" in capsys.readouterr().err
    assert _texto(docs) == antes


# --- flujo completo y parser ------------------------------------------------------------


def test_flujo_completo_de_un_experimento(docs, escribir_json, tmp_path):
    runs_whole = _runs(tmp_path, escribir_json, "runs_whole", n=3)
    runs_owl = _runs(
        tmp_path, escribir_json, "runs_owl", n=4,
        env={"detector": "owlvit"}, raiz={"model_load_ms": 8500.0},
    )  # fmt: skip

    assert ejecutar(docs, "nuevo", "--objetivo", "validar el pipeline de la v0",
                    "--dataset", "fotos propias de prueba, sin etiquetas") == 0  # fmt: skip
    for runs in (runs_whole, runs_owl):
        assert _ejecucion(docs, runs) == 0
    assert ejecutar(docs, "campos", "--exp", "exp001", "--set", "estado=hecho") == 0

    cabecera = _cabecera(docs)
    assert cabecera["estado"] == "hecho" and cabecera["commit"] == "abc1234"
    assert cabecera["objetivo"] == "validar el pipeline de la v0"
    filas = [(f["Carpeta"], f["Detector"], f["Imágenes"]) for f in _filas(docs)]
    assert filas == [("runs_whole", "whole", "3"), ("runs_owl", "owlvit", "4")]
    assert _filas(docs)[1]["Carga modelo (ms)"] == "8500.0"


def test_la_orden_resumen_ya_no_existe(docs):
    with pytest.raises(SystemExit) as salida:
        ejecutar(docs, "resumen")

    assert salida.value.code == 2  # argparse: orden desconocida
    assert not (docs.raiz / "informes").exists()


def test_la_carpeta_docs_por_defecto_es_la_del_repositorio():
    argumentos = crear_parser().parse_args(["campos", "--exp", "exp001", "--set", "estado=hecho"])

    assert argumentos.docs == RAIZ / "docs"
