"""Pruebas de la evaluación completa con métodos simulados (sin modelos reales ni GPU).

Cifras esperadas, calculadas a mano con las predicciones de ``conftest.py``:

* Método con confianza (umbral 0,5): imagen a -> 2 TP y 1 FP (el duplicado; la caja de confianza
  0,3 queda bajo el umbral); b -> sin predicciones, 1 FN; c -> 1 FP y 1 FN.
  Total TP=2, FP=2, FN=2: precisión 0,5, recall 0,5, F1 0,5; 1 imagen sin detección.
* Método sin confianza: a -> 2 TP; b -> 1 TP; c -> 1 FN. TP=3, FP=0, FN=1: precisión 1, recall
  0,75, F1 6/7; 1 imagen sin detección.
* Latencias simuladas: 10, 20 y 30 ms por imagen -> mediana 20 ms y p90 28 ms.
"""

import csv
import dataclasses
import json
from pathlib import Path

import pytest
import yaml

from src.config import load_config
from src.eval.conjunto import cargar_conjunto_coco
from src.eval.localizacion import ejecutar, evaluar_metodo, main
from src.experimentos.privacidad import contiene_ruta_absoluta, contiene_usuario

pytest.importorskip("pycocotools")

RAIZ = Path(__file__).resolve().parents[2]
COLUMNAS = [
    "metodo", "imagen", "n_gt", "n_pred", "tp", "fp", "fn", "precision", "recall", "f1",
    "iou_medio", "sin_deteccion", "latencia_ms",
]  # fmt: skip


@pytest.fixture
def resumen(config_sintetica, raiz_sintetica, fabricas, entorno_falso, reloj):
    return ejecutar(
        config_sintetica, "prueba_001", raiz=raiz_sintetica, fabricas=fabricas,
        entorno=entorno_falso, reloj=reloj,
    )  # fmt: skip


def _carpeta(raiz):
    return raiz / "runs" / "prueba_001"


def _filas(raiz):
    with (_carpeta(raiz) / "por_imagen.csv").open(encoding="utf-8", newline="") as archivo:
        return list(csv.DictReader(archivo))


def test_se_crean_todas_las_salidas(resumen, raiz_sintetica):
    carpeta = _carpeta(raiz_sintetica)
    for nombre in ("por_imagen.csv", "resumen.json", "tabla.md"):
        assert (carpeta / nombre).is_file()
    for metodo in ("owlvit", "rfdetr", "linea_base_hsv"):
        ejemplos = sorted(p.name for p in (carpeta / "ejemplos" / metodo).iterdir())
        assert len(ejemplos) == 2  # ejemplos.numero de la configuración de la prueba
    nombres = {
        m: sorted(p.name for p in (carpeta / "ejemplos" / m).iterdir())
        for m in ("owlvit", "rfdetr", "linea_base_hsv")
    }
    assert nombres["owlvit"] == nombres["rfdetr"] == nombres["linea_base_hsv"]  # mismas imágenes


def test_el_csv_tiene_una_fila_por_imagen_y_metodo(resumen, raiz_sintetica):
    ruta = _carpeta(raiz_sintetica) / "por_imagen.csv"
    cabecera = ruta.read_text(encoding="utf-8").splitlines()[0]
    assert cabecera.split(",") == COLUMNAS
    filas = _filas(raiz_sintetica)
    assert len(filas) == 9
    a = next(f for f in filas if f["metodo"] == "owlvit" and f["imagen"] == "a_ñ.png")
    assert (a["n_gt"], a["n_pred"], a["tp"], a["fp"], a["fn"]) == ("2", "3", "2", "1", "0")
    assert float(a["iou_medio"]) == pytest.approx(1.0)
    assert a["sin_deteccion"] == "0"
    assert float(a["latencia_ms"]) == pytest.approx(10.0)
    b = next(f for f in filas if f["metodo"] == "owlvit" and f["imagen"] == "b.png")
    assert b["precision"] == "" and b["iou_medio"] == ""  # sin definir: vacío, no 0
    assert float(b["recall"]) == 0.0 and b["sin_deteccion"] == "1"


def test_metricas_globales_del_metodo_con_confianza(resumen):
    m = resumen["metodos"]["owlvit"]
    assert (m["metricas"]["tp"], m["metricas"]["fp"], m["metricas"]["fn"]) == (2, 2, 2)
    assert m["metricas"]["precision"] == pytest.approx(0.5)
    assert m["metricas"]["recall"] == pytest.approx(0.5)
    assert m["metricas"]["f1"] == pytest.approx(0.5)
    assert m["metricas"]["iou_medio"] == pytest.approx(1.0)
    assert m["imagenes_sin_deteccion"] == 1
    assert 0 < m["ap50"] <= 1
    assert 0 <= m["ap50_95"] <= m["ap50"] + 1e-9  # aquí todas las cajas aciertan con IoU 1


def test_la_linea_base_no_tiene_ap(resumen):
    m = resumen["metodos"]["linea_base_hsv"]
    assert (m["metricas"]["tp"], m["metricas"]["fp"], m["metricas"]["fn"]) == (3, 0, 1)
    assert m["metricas"]["precision"] == pytest.approx(1.0)
    assert m["metricas"]["recall"] == pytest.approx(0.75)
    assert m["metricas"]["f1"] == pytest.approx(6 / 7)
    assert m["imagenes_sin_deteccion"] == 1
    assert m["ap50"] is None and m["ap50_95"] is None


def test_latencias_y_carga(resumen):
    m = resumen["metodos"]["owlvit"]
    assert m["latencia"]["mediana_ms"] == pytest.approx(20.0)
    assert m["latencia"]["p90_ms"] == pytest.approx(28.0)
    assert m["latencia"]["repeticiones"] == 2 and m["latencia"]["calentamiento"] == 1
    assert m["carga_modelo_s"] == 0.5


def test_el_resumen_json_coincide_con_lo_devuelto_y_lleva_el_entorno(resumen, raiz_sintetica):
    en_disco = json.loads((_carpeta(raiz_sintetica) / "resumen.json").read_text(encoding="utf-8"))
    assert en_disco == resumen
    assert en_disco["entorno"]["commit"] == "abc1234" and en_disco["entorno"]["dirty"] is False
    assert en_disco["exp_id"] == "prueba_001"
    assert en_disco["conjunto"] == {
        "carpeta": "conjunto", "particion": "valid", "imagenes": 3, "cajas_referencia": 4
    }
    assert en_disco["iou_min"] == 0.5


def test_la_tabla_lista_para_la_memoria(resumen, raiz_sintetica):
    tabla = (_carpeta(raiz_sintetica) / "tabla.md").read_text(encoding="utf-8")
    assert tabla.count("|") > 0
    for nombre in ("owlvit", "rfdetr", "linea_base_hsv"):
        assert nombre in tabla
    linea = next(f for f in tabla.splitlines() if f.startswith("| linea_base_hsv"))
    assert "n/a" in linea  # AP no aplica
    assert "0,750" in linea  # coma decimal
    assert "20,0" in next(f for f in tabla.splitlines() if f.startswith("| owlvit"))
    # Avisos fijos de interpretación:
    assert "no están ajustados" in tabla  # parámetros HSV
    assert "fijado de antemano" in tabla  # umbral de RF-DETR
    assert "elegir el checkpoint" in tabla  # valid sesga a RF-DETR
    assert "p90" in tabla and "medianas por imagen" in tabla


def test_la_tabla_anota_los_valores_usados(resumen, raiz_sintetica):
    tabla = (_carpeta(raiz_sintetica) / "tabla.md").read_text(encoding="utf-8")
    assert "detalle" in tabla  # los parámetros de cada método
    assert "IoU >= 0,5" in tabla


def test_ninguna_salida_lleva_ruta_absoluta_ni_usuario(resumen, raiz_sintetica):
    carpeta = _carpeta(raiz_sintetica)
    textos = {".csv", ".json", ".md"}
    archivos = [p for p in carpeta.rglob("*") if p.is_file() and p.suffix in textos]
    assert len(archivos) == 3
    for archivo in archivos:
        texto = archivo.read_text(encoding="utf-8")
        assert not contiene_ruta_absoluta(texto), archivo.name
        assert not contiene_usuario(texto), archivo.name
        assert str(raiz_sintetica) not in texto and raiz_sintetica.as_posix() not in texto


def test_no_pisa_una_ejecucion_existente(resumen, config_sintetica, raiz_sintetica, fabricas):
    with pytest.raises(ValueError, match="ya existe"):
        ejecutar(config_sintetica, "prueba_001", raiz=raiz_sintetica, fabricas=fabricas)


@pytest.mark.parametrize("malo", ["", "con espacio", "../fuera", "a/b", ".oculto"])
def test_exp_id_no_valido(malo, config_sintetica, raiz_sintetica, fabricas):
    with pytest.raises(ValueError, match="exp-id"):
        ejecutar(config_sintetica, malo, raiz=raiz_sintetica, fabricas=fabricas)


def test_solo_se_ejecutan_los_metodos_activos(config_sintetica, raiz_sintetica, fabricas, reloj,
                                              entorno_falso):  # fmt: skip
    cfg = dataclasses.replace(config_sintetica, metodos=("linea_base_hsv",))
    resumen = ejecutar(cfg, "solo_base", raiz=raiz_sintetica, fabricas=fabricas,
                       entorno=entorno_falso, reloj=reloj)  # fmt: skip
    assert list(resumen["metodos"]) == ["linea_base_hsv"]
    assert not (raiz_sintetica / "runs" / "solo_base" / "ejemplos" / "owlvit").exists()


def test_tamano_de_imagen_distinto_al_del_json(config_sintetica, raiz_sintetica, fabricas, reloj):
    ruta = raiz_sintetica / "conjunto" / "valid" / "_annotations.coco.json"
    datos = json.loads(ruta.read_text(encoding="utf-8"))
    datos["images"][0]["width"] = 999
    ruta.write_text(json.dumps(datos), encoding="utf-8")
    with pytest.raises(ValueError, match="tamaño"):
        ejecutar(config_sintetica, "mal_tamano", raiz=raiz_sintetica, fabricas=fabricas,
                 reloj=reloj)  # fmt: skip


def test_evaluar_metodo_filtra_por_el_umbral_operativo_pero_conserva_las_del_ap(
    config_sintetica, raiz_sintetica, fabricar_metodo, reloj
):
    imagenes = cargar_conjunto_coco(raiz_sintetica / "conjunto" / "valid", "lechuga")
    resultado = evaluar_metodo(fabricar_metodo(True), imagenes, config_sintetica, reloj=reloj)
    a = resultado.resultados[0]
    assert a.n_pred == 3  # 0,9, 0,8 y 0,7; la de 0,3 no supera el umbral operativo
    assert len(resultado.predicciones_ap[0]) == 4  # pero el AP sí la ve
    assert len(resultado.predicciones_operativas[0]) == 3


# --- Línea de órdenes ----------------------------------------------------------------------


def _yaml_sintetico(tmp_path):
    datos = load_config(RAIZ / "configs" / "eval" / "localizacion.yaml")
    datos["conjunto"]["carpeta"] = "conjunto"
    datos["medicion"] = {"repeticiones": 2, "calentamiento": 1}
    datos["ejemplos"]["numero"] = 1
    ruta = tmp_path / "loc.yaml"
    ruta.write_text(yaml.safe_dump(datos, allow_unicode=True), encoding="utf-8")
    return ruta


def test_main_ejecuta_y_devuelve_cero(raiz_sintetica, fabricas, entorno_falso, reloj, capsys):
    ruta = _yaml_sintetico(raiz_sintetica)
    codigo = main(
        ["--config", str(ruta), "--exp-id", "cli_001", "--raiz", str(raiz_sintetica)],
        fabricas=fabricas, entorno=entorno_falso, reloj=reloj,
    )  # fmt: skip
    assert codigo == 0
    assert (raiz_sintetica / "runs" / "cli_001" / "tabla.md").is_file()
    assert str(raiz_sintetica) not in capsys.readouterr().out


def test_main_con_configuracion_inexistente_devuelve_uno(raiz_sintetica, capsys):
    codigo = main(["--config", str(raiz_sintetica / "no.yaml"), "--exp-id", "x",
                   "--raiz", str(raiz_sintetica)])  # fmt: skip
    capturado = capsys.readouterr()
    assert codigo == 1
    assert "ERROR" in capturado.err
    assert str(raiz_sintetica) not in capturado.err  # el mensaje sale saneado


def test_main_con_configuracion_invalida_devuelve_uno(raiz_sintetica, capsys):
    ruta = raiz_sintetica / "mala.yaml"
    ruta.write_text("iou_min: 3\n", encoding="utf-8")
    codigo = main(["--config", str(ruta), "--exp-id", "x", "--raiz", str(raiz_sintetica)])
    assert codigo == 1
    assert "ERROR" in capsys.readouterr().err


def test_main_exige_exp_id():
    with pytest.raises(SystemExit):
        main(["--config", "configs/eval/localizacion.yaml"])
