"""Pruebas de src/train/config.py: lectura y validación del YAML de entrenamiento."""

from pathlib import Path

import pytest
import yaml

from src.train.config import MODELOS_APACHE, cargar_config_entrenamiento

RAIZ = Path(__file__).resolve().parents[2]


# Todas las claves del YAML (con puntos para las anidadas). Una prueba comprueba que esta lista
# coincide con la configuración válida de las fixtures, para que no se quede atrás.
CLAVES = [
    "modelo.clase", "modelo.resolucion", "modelo.clases",
    "entrenamiento.epocas", "entrenamiento.batch_size", "entrenamiento.grad_accum_steps",
    "entrenamiento.lr", "entrenamiento.lr_encoder", "entrenamiento.amp_dtype",
    "entrenamiento.num_workers", "entrenamiento.early_stopping",
    "entrenamiento.checkpoint_interval", "entrenamiento.tensorboard",
    "entrenamiento.dispositivo",
    "semilla",
    "datos.dataset", "datos.repeticiones_svodk", "datos.prefijo_svodk", "datos.trabajo",
    "salida.carpeta",
    "smoke.epocas", "smoke.imagenes_train",
]  # fmt: skip


def _claves_hoja(config: dict, prefijo: str = "") -> list[str]:
    salida = []
    for clave, valor in config.items():
        ruta = f"{prefijo}{clave}"
        salida += _claves_hoja(valor, ruta + ".") if isinstance(valor, dict) else [ruta]
    return salida


def test_la_lista_de_claves_cubre_toda_la_configuracion_valida(config_valida):
    assert sorted(_claves_hoja(config_valida)) == sorted(CLAVES)


def test_un_yaml_valido_se_lee_con_todos_sus_valores(escribir_config):
    config = cargar_config_entrenamiento(escribir_config())

    assert config.modelo_clase == "RFDETRNano"
    assert config.resolucion == 384 and config.clases == ("lechuga",)
    assert config.epocas == 30 and config.batch_size == 8 and config.grad_accum_steps == 2
    assert config.lr == 1.0e-4 and config.lr_encoder == 1.5e-4
    assert config.amp_dtype == "auto" and config.dispositivo == "auto"
    assert config.num_workers == 2 and config.early_stopping is False
    assert config.checkpoint_interval == 10 and config.tensorboard is False
    assert config.semilla == 42
    assert config.dataset == "data/processed/etapa1_v1"
    assert config.repeticiones_svodk == 1 and config.prefijo_svodk == "svodk_"
    assert config.trabajo == "data/processed/trabajo" and config.salida == "runs"
    assert config.smoke_epocas == 1 and config.smoke_imagenes_train == 4


def test_a_diccionario_devuelve_la_misma_forma_que_el_yaml(escribir_config, config_valida):
    config = cargar_config_entrenamiento(escribir_config())

    assert config.a_diccionario() == config_valida


@pytest.mark.parametrize("clave", CLAVES)
def test_si_falta_una_clave_el_error_la_nombra(escribir_config, clave):
    with pytest.raises(ValueError, match=clave.replace(".", r"\.")):
        cargar_config_entrenamiento(escribir_config(quitar=(clave,)))


@pytest.mark.parametrize("seccion", ["modelo", "entrenamiento", "datos", "salida", "smoke"])
def test_si_falta_una_seccion_entera_el_error_la_nombra(config_valida, tmp_path, seccion):
    config = config_valida
    del config[seccion]
    ruta = tmp_path / "sin_seccion.yaml"
    ruta.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValueError, match=seccion):
        cargar_config_entrenamiento(ruta)


@pytest.mark.parametrize(
    ("clave", "valor"),
    [
        ("entrenamiento.epocas", "diez"),
        ("entrenamiento.epocas", 0),
        ("entrenamiento.epocas", True),
        ("entrenamiento.epocas", 2.5),
        ("entrenamiento.batch_size", -1),
        ("entrenamiento.grad_accum_steps", 0),
        ("entrenamiento.lr", 0),
        ("entrenamiento.lr", "alto"),
        ("entrenamiento.lr_encoder", -1.0e-4),
        ("entrenamiento.amp_dtype", "fp64"),
        ("entrenamiento.dispositivo", "tpu"),
        ("entrenamiento.num_workers", -1),
        ("entrenamiento.early_stopping", "no"),
        ("entrenamiento.checkpoint_interval", 0),
        ("entrenamiento.tensorboard", 1),
        ("semilla", -1),
        ("semilla", 1.5),
        ("modelo.resolucion", 0),
        ("modelo.resolucion", "384"),
        ("modelo.clases", []),
        ("modelo.clases", "lechuga"),
        ("modelo.clases", [1, 2]),
        ("datos.repeticiones_svodk", 0),
        ("datos.repeticiones_svodk", -2),
        ("datos.repeticiones_svodk", 1.5),
        ("datos.repeticiones_svodk", "2"),
        ("datos.repeticiones_svodk", True),
        ("datos.prefijo_svodk", ""),
        ("datos.dataset", ""),
        ("datos.trabajo", ""),
        ("salida.carpeta", ""),
        ("smoke.epocas", 0),
        ("smoke.imagenes_train", 0),
    ],
)
def test_un_valor_no_valido_da_error_que_nombra_la_clave(escribir_config, clave, valor):
    with pytest.raises(ValueError, match=clave.replace(".", r"\.")):
        cargar_config_entrenamiento(escribir_config({clave: valor}))


def test_una_clave_desconocida_da_error_como_haria_rfdetr_con_un_nombre_mal_escrito(
    escribir_config,
):
    with pytest.raises(ValueError, match=r"entrenamiento\.epocass"):
        cargar_config_entrenamiento(escribir_config({"entrenamiento.epocass": 3}))


@pytest.mark.parametrize("modelo", ["RFDETRXLarge", "RFDETR2XLarge"])
def test_los_modelos_de_rfdetr_plus_se_rechazan_por_licencia(escribir_config, modelo):
    with pytest.raises(ValueError, match="rfdetr_plus") as error:
        cargar_config_entrenamiento(escribir_config({"modelo.clase": modelo}))

    assert modelo in str(error.value)


def test_un_modelo_desconocido_da_error_con_los_validos(escribir_config):
    with pytest.raises(ValueError, match="RFDETRNano"):
        cargar_config_entrenamiento(escribir_config({"modelo.clase": "ModeloInventado"}))


def test_solo_se_admiten_modelos_apache():
    assert MODELOS_APACHE == ("RFDETRNano", "RFDETRSmall", "RFDETRMedium")


def test_un_archivo_inexistente_da_error_claro(tmp_path):
    with pytest.raises(FileNotFoundError, match="no_existe.yaml"):
        cargar_config_entrenamiento(tmp_path / "no_existe.yaml")


def test_un_yaml_que_no_es_un_diccionario_da_error(tmp_path):
    ruta = tmp_path / "lista.yaml"
    ruta.write_text("- 1\n- 2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="diccionario"):
        cargar_config_entrenamiento(ruta)


def test_el_yaml_real_del_repositorio_es_valido_y_tiene_los_valores_acordados():
    config = cargar_config_entrenamiento(RAIZ / "configs" / "train" / "etapa1_rfdetr_n.yaml")

    assert config.modelo_clase == "RFDETRNano" and config.resolucion == 384
    assert config.epocas == 30 and config.batch_size == 8 and config.grad_accum_steps == 2
    assert config.lr == 1.0e-4 and config.lr_encoder == 1.5e-4 and config.semilla == 42
    assert config.repeticiones_svodk == 1 and config.prefijo_svodk == "svodk_"
    assert config.dataset == "data/processed/etapa1_v1"
    assert config.trabajo == "data/processed/trabajo" and config.salida == "runs"
    assert config.smoke_epocas == 1 and config.smoke_imagenes_train == 20


def test_ningun_valor_del_yaml_real_es_una_ruta_absoluta():
    texto = (RAIZ / "configs" / "train" / "etapa1_rfdetr_n.yaml").read_text(encoding="utf-8")

    assert "C:\\" not in texto and "/content" not in texto and "/home/" not in texto
