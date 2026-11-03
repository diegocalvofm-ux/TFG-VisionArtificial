"""Pruebas de la lectura y validación de configs/eval/localizacion.yaml."""

import copy
from pathlib import Path

import pytest
import yaml

from src.config import load_config
from src.eval.config import METODOS, ParametrosHSV, cargar_config_localizacion

RAIZ = Path(__file__).resolve().parents[2]
YAML_REAL = RAIZ / "configs" / "eval" / "localizacion.yaml"


def _escribir(tmp_path, cambiar=None):
    """Copia el YAML real a tmp_path aplicando ``cambiar(dict)`` y devuelve la ruta."""
    datos = copy.deepcopy(load_config(YAML_REAL))
    if cambiar:
        cambiar(datos)
    ruta = tmp_path / "loc.yaml"
    ruta.write_text(yaml.safe_dump(datos, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return ruta


def test_el_yaml_real_carga_con_los_valores_acordados():
    c = cargar_config_localizacion(YAML_REAL)
    assert c.metodos == METODOS == ("owlvit", "rfdetr", "linea_base_hsv")
    assert c.conjunto_carpeta == "data/processed/etapa1_v1"
    assert c.particion == "valid"
    assert c.clase == "lechuga"
    assert c.iou_min == 0.5
    assert c.dispositivo == "cpu"
    assert (c.repeticiones, c.calentamiento) == (5, 1)
    assert c.owlvit_max_detecciones == 50  # v0 usa 10; aquí no se topa el recall
    assert c.owlvit_consulta is None and c.owlvit_umbral is None  # los de configs/v0.yaml
    assert c.rfdetr_umbral == 0.5
    assert c.rfdetr_pesos == "runs/exp002-etapa1-rfdert-n/pesos/checkpoint_best_total.pth"
    assert c.ejemplos_numero == 10
    assert c.salida == "runs"
    assert c.hsv == ParametrosHSV(
        h=(35, 85), s=(40, 255), v=(40, 255), apertura_px=5, cierre_px=15, area_min_frac=0.005
    )


def test_la_configuracion_se_puede_volver_a_escribir_igual_que_el_yaml():
    c = cargar_config_localizacion(YAML_REAL)
    assert c.a_diccionario() == load_config(YAML_REAL)


def test_el_yaml_real_no_cambia_configs_v0():
    # Se pidió no tocar configs/v0.yaml: sigue con el tope de 10 detecciones.
    assert load_config(RAIZ / "configs" / "v0.yaml")["owlvit"]["max_detecciones"] == 10


def test_metodos_subconjunto(tmp_path):
    c = cargar_config_localizacion(_escribir(tmp_path, lambda d: d.update(metodos=["rfdetr"])))
    assert c.metodos == ("rfdetr",)


def test_archivo_inexistente(tmp_path):
    with pytest.raises(FileNotFoundError, match="configuración"):
        cargar_config_localizacion(tmp_path / "no_existe.yaml")


def test_yaml_vacio(tmp_path):
    ruta = tmp_path / "vacio.yaml"
    ruta.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="diccionario"):
        cargar_config_localizacion(ruta)


def test_yaml_mal_escrito(tmp_path):
    ruta = tmp_path / "roto.yaml"
    ruta.write_text("conjunto: [sin cerrar\n", encoding="utf-8")
    with pytest.raises(yaml.YAMLError):
        cargar_config_localizacion(ruta)


@pytest.mark.parametrize(
    "cambiar, clave",
    [
        (lambda d: d.pop("iou_min"), "iou_min"),
        (lambda d: d["conjunto"].pop("particion"), "conjunto.particion"),
        (lambda d: d["owlvit"].pop("umbral_ap"), "owlvit.umbral_ap"),
        (lambda d: d["rfdetr"].pop("pesos"), "rfdetr.pesos"),
        (lambda d: d["linea_base_hsv"].pop("h"), "linea_base_hsv.h"),
        (lambda d: d.pop("ejemplos"), "ejemplos"),
    ],
)
def test_falta_una_clave(tmp_path, cambiar, clave):
    with pytest.raises(ValueError, match=f"falta '{clave}'"):
        cargar_config_localizacion(_escribir(tmp_path, cambiar))


def test_clave_desconocida(tmp_path):
    ruta = _escribir(tmp_path, lambda d: d["rfdetr"].update(umbrall=0.5))
    with pytest.raises(ValueError, match="rfdetr.umbrall"):
        cargar_config_localizacion(ruta)


def test_una_seccion_que_no_es_un_bloque(tmp_path):
    with pytest.raises(ValueError, match="'owlvit' debe ser un bloque"):
        cargar_config_localizacion(_escribir(tmp_path, lambda d: d.update(owlvit=3)))


@pytest.mark.parametrize(
    "cambiar, clave",
    [
        (lambda d: d.update(iou_min=0), "iou_min"),
        (lambda d: d.update(iou_min=1.5), "iou_min"),
        (lambda d: d.update(iou_min="alto"), "iou_min"),
        (lambda d: d.update(dispositivo="tpu"), "dispositivo"),
        (lambda d: d["medicion"].update(repeticiones=0), "medicion.repeticiones"),
        (lambda d: d["medicion"].update(calentamiento=-1), "medicion.calentamiento"),
        (lambda d: d["medicion"].update(repeticiones=True), "medicion.repeticiones"),
        (lambda d: d["owlvit"].update(umbral=1.2), "owlvit.umbral"),
        (lambda d: d["owlvit"].update(max_detecciones=0), "owlvit.max_detecciones"),
        (lambda d: d["owlvit"].update(consulta=""), "owlvit.consulta"),
        (lambda d: d["rfdetr"].update(umbral=-0.1), "rfdetr.umbral"),
        (lambda d: d["rfdetr"].update(pesos=""), "rfdetr.pesos"),
        (lambda d: d["linea_base_hsv"].update(h=[85, 35]), "linea_base_hsv.h"),
        (lambda d: d["linea_base_hsv"].update(h=[0, 200]), "linea_base_hsv.h"),
        (lambda d: d["linea_base_hsv"].update(s=[10]), "linea_base_hsv.s"),
        (lambda d: d["linea_base_hsv"].update(apertura_px=0), "linea_base_hsv.apertura_px"),
        (lambda d: d["linea_base_hsv"].update(area_min_frac=0), "linea_base_hsv.area_min_frac"),
        (lambda d: d["linea_base_hsv"].update(area_min_frac=1), "linea_base_hsv.area_min_frac"),
        (lambda d: d["ejemplos"].update(numero=-1), "ejemplos.numero"),
    ],
)
def test_valor_invalido_nombra_la_clave(tmp_path, cambiar, clave):
    with pytest.raises(ValueError, match=clave.replace(".", r"\.")):
        cargar_config_localizacion(_escribir(tmp_path, cambiar))


def test_metodo_desconocido(tmp_path):
    ruta = _escribir(tmp_path, lambda d: d.update(metodos=["owlvit", "yolo"]))
    with pytest.raises(ValueError, match="yolo"):
        cargar_config_localizacion(ruta)


def test_metodos_vacio_o_repetido(tmp_path):
    with pytest.raises(ValueError, match="metodos"):
        cargar_config_localizacion(_escribir(tmp_path, lambda d: d.update(metodos=[])))
    with pytest.raises(ValueError, match="repet"):
        cargar_config_localizacion(
            _escribir(tmp_path, lambda d: d.update(metodos=["rfdetr", "rfdetr"]))
        )


def test_umbral_operativo_menor_que_el_de_ap(tmp_path):
    ruta = _escribir(tmp_path, lambda d: d["rfdetr"].update(umbral=0.0005))
    with pytest.raises(ValueError, match="rfdetr.umbral"):
        cargar_config_localizacion(ruta)


def test_las_rutas_deben_ser_relativas(tmp_path):
    ruta = _escribir(tmp_path, lambda d: d["rfdetr"].update(pesos="C:/Users/x/pesos.pth"))
    with pytest.raises(ValueError, match="relativa"):
        cargar_config_localizacion(ruta)
    ruta = _escribir(tmp_path, lambda d: d["conjunto"].update(carpeta="/home/x/datos"))
    with pytest.raises(ValueError, match="relativa"):
        cargar_config_localizacion(ruta)
