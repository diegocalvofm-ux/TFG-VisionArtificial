"""Pruebas de src/train/etapa2/config.py: lectura y validación del YAML de la etapa 2."""

from pathlib import Path

import pytest

from src.train.etapa2.config import ARQUITECTURAS, cargar_config_etapa2

RAIZ = Path(__file__).resolve().parents[3]
YAML_REPO = RAIZ / "configs" / "train" / "etapa2_effb0.yaml"

# Todas las claves del YAML (con puntos para las anidadas). Una prueba comprueba que esta lista
# coincide con la configuración válida de las fixtures, para que no se quede atrás.
CLAVES = [
    "modelo.arquitectura", "modelo.preentrenado", "modelo.resolucion", "modelo.dropout",
    "entrenamiento.epocas", "entrenamiento.batch_size", "entrenamiento.lr_extractor",
    "entrenamiento.lr_cabezas", "entrenamiento.decaimiento_pesos",
    "entrenamiento.epocas_calentamiento", "entrenamiento.factor_lr_final",
    "entrenamiento.precision_mixta", "entrenamiento.num_workers", "entrenamiento.dispositivo",
    "perdida.pesos_clase",
    "aumentos.escala_recorte", "aumentos.proporcion_recorte", "aumentos.volteo_horizontal",
    "aumentos.volteo_vertical", "aumentos.giros_90", "aumentos.rotacion_grados",
    "aumentos.brillo", "aumentos.contraste",
    "semilla",
    "datos.dataset", "datos.fuente_propia", "datos.sobremuestreo_propio", "datos.reparto_propio",
    "salida.carpeta",
    "smoke.epocas", "smoke.recortes_train", "smoke.recortes_eval",
]  # fmt: skip


def _claves_hoja(config: dict, prefijo: str = "") -> list[str]:
    salida = []
    for clave, valor in config.items():
        ruta = f"{prefijo}{clave}"
        salida += _claves_hoja(valor, ruta + ".") if isinstance(valor, dict) else [ruta]
    return salida


def test_la_lista_de_claves_cubre_toda_la_configuracion_valida(config_valida):
    assert sorted(_claves_hoja(config_valida)) == sorted(CLAVES)


def test_el_yaml_del_repositorio_es_valido_y_tiene_los_valores_acordados():
    config = cargar_config_etapa2(YAML_REPO)

    assert config.arquitectura == "efficientnet_b0" and config.preentrenado is True
    assert config.resolucion == 320
    # Decisiones del autor: recorte sin deformar y sin perder más del 10 % del área.
    assert config.aumentos.escala_recorte == (0.9, 1.0)
    assert config.aumentos.proporcion_recorte == (1.0, 1.0)
    assert config.aumentos.brillo == 0 and config.aumentos.contraste == 0
    assert config.pesos_clase == "inv_raiz"
    assert config.dataset == "data/processed/etapa2_v1" and config.fuente_propia == "propio"
    assert config.reparto_propio is None
    assert (config.smoke_epocas, config.smoke_recortes_train, config.smoke_recortes_eval) == (
        1, 64, 64
    )  # fmt: skip


def test_un_yaml_valido_se_lee_con_todos_sus_valores(escribir_config):
    config = cargar_config_etapa2(escribir_config())

    assert (config.arquitectura, config.preentrenado, config.resolucion, config.dropout) == (
        "efficientnet_b0", False, 32, 0.2
    )  # fmt: skip
    assert (config.epocas, config.batch_size) == (2, 4)
    assert (config.lr_extractor, config.lr_cabezas, config.decaimiento_pesos) == (
        1.0e-3, 1.0e-2, 1.0e-2
    )  # fmt: skip
    assert (config.epocas_calentamiento, config.factor_lr_final) == (1, 0.01)
    assert config.precision_mixta is True and config.num_workers == 0
    assert config.dispositivo == "cpu" and config.semilla == 42
    assert config.aumentos.volteo_horizontal and config.aumentos.volteo_vertical
    assert config.aumentos.giros_90 and config.aumentos.rotacion_grados == 10
    assert config.sobremuestreo_propio == 1 and config.salida == "runs"


@pytest.mark.parametrize("clave", CLAVES)
def test_si_falta_una_clave_el_error_la_nombra(escribir_config, clave):
    with pytest.raises(ValueError, match=f"falta '{clave}'"):
        cargar_config_etapa2(escribir_config(sin=(clave,)))


def test_una_clave_desconocida_se_rechaza_y_se_nombra(escribir_config):
    ruta = escribir_config({"entrenamiento.epcas": 3, "extra": 1})

    with pytest.raises(ValueError, match="claves desconocidas") as error:
        cargar_config_etapa2(ruta)

    assert "entrenamiento.epcas" in str(error.value) and "extra" in str(error.value)


@pytest.mark.parametrize("clave", ["aumentos.saturacion", "aumentos.tono", "aumentos.hue"])
def test_no_se_pueden_pedir_cambios_de_tono_ni_de_saturacion(escribir_config, clave):
    with pytest.raises(ValueError, match="tono ni de saturación") as error:
        cargar_config_etapa2(escribir_config({clave: 0.1}))

    assert clave in str(error.value)


@pytest.mark.parametrize(
    ("clave", "valor"),
    [
        ("modelo.arquitectura", "resnet50"),
        ("modelo.preentrenado", "si"),
        ("modelo.resolucion", 16),
        ("modelo.dropout", 1.0),
        ("modelo.dropout", -0.1),
        ("entrenamiento.epocas", 0),
        ("entrenamiento.batch_size", 0),
        ("entrenamiento.lr_extractor", 0),
        ("entrenamiento.lr_cabezas", -1.0e-3),
        ("entrenamiento.decaimiento_pesos", -0.1),
        ("entrenamiento.epocas_calentamiento", -1),
        ("entrenamiento.factor_lr_final", 1.5),
        ("entrenamiento.precision_mixta", 1),
        ("entrenamiento.num_workers", -1),
        ("entrenamiento.dispositivo", "tpu"),
        ("perdida.pesos_clase", "log"),
        ("aumentos.escala_recorte", [0.0, 1.0]),
        ("aumentos.escala_recorte", [0.9, 1.1]),
        ("aumentos.escala_recorte", [1.0, 0.9]),
        ("aumentos.escala_recorte", 0.9),
        ("aumentos.escala_recorte", [0.9, 1.0, 1.0]),
        ("aumentos.proporcion_recorte", [1.2, 1.0]),
        ("aumentos.proporcion_recorte", [0, 1.0]),
        ("aumentos.volteo_vertical", "no"),
        ("aumentos.giros_90", 1),
        ("aumentos.rotacion_grados", 45),
        ("aumentos.brillo", 0.3),
        ("aumentos.contraste", -0.1),
        ("datos.dataset", "/abs/etapa2_v1"),
        ("datos.dataset", "C:/datos/etapa2_v1"),
        ("datos.fuente_propia", ""),
        ("datos.sobremuestreo_propio", 0),
        ("datos.sobremuestreo_propio", 2.5),
        ("semilla", -1),
        ("salida.carpeta", ""),
        ("smoke.epocas", 0),
        ("smoke.recortes_train", 0),
        ("smoke.recortes_eval", True),
    ],
)
def test_un_valor_no_valido_da_un_error_que_nombra_la_clave(escribir_config, clave, valor):
    with pytest.raises(ValueError, match=clave.replace(".", r"\.")):
        cargar_config_etapa2(escribir_config({clave: valor}))


def test_una_arquitectura_no_permitida_dice_cuales_valen(escribir_config):
    with pytest.raises(ValueError, match="efficientnet_b0"):
        cargar_config_etapa2(escribir_config({"modelo.arquitectura": "vit_base_patch16_224"}))


def test_las_arquitecturas_permitidas_llevan_su_licencia():
    assert "efficientnet_b0" in ARQUITECTURAS
    assert all("Apache-2.0" in descripcion for descripcion in ARQUITECTURAS.values())


def test_reparto_propio_valido_se_lee_como_tres_fracciones(escribir_config):
    reparto = {"train": 0.7, "valid": 0.15, "test": 0.15}

    config = cargar_config_etapa2(escribir_config({"datos.reparto_propio": reparto}))

    assert config.reparto_propio == (0.7, 0.15, 0.15)


@pytest.mark.parametrize(
    "valor",
    [
        {"train": 0.7, "valid": 0.3},
        {"train": 0.7, "val": 0.15, "test": 0.15},
        {"train": 0.5, "valid": 0.3, "test": 0.3},
        {"train": 0.7, "valid": 0, "test": 0.3},
        {"train": "mucho", "valid": 0.15, "test": 0.15},
        0.7,
    ],
)
def test_reparto_propio_mal_escrito_da_error(escribir_config, valor):
    with pytest.raises(ValueError, match=r"datos\.reparto_propio"):
        cargar_config_etapa2(escribir_config({"datos.reparto_propio": valor}))


def test_a_diccionario_tiene_la_forma_del_yaml(escribir_config, config_valida):
    assert cargar_config_etapa2(escribir_config()).a_diccionario() == config_valida


def test_a_diccionario_con_reparto_lo_escribe_por_particion(escribir_config):
    reparto = {"train": 0.6, "valid": 0.2, "test": 0.2}

    config = cargar_config_etapa2(escribir_config({"datos.reparto_propio": reparto}))

    assert config.a_diccionario()["datos"]["reparto_propio"] == reparto


def test_un_archivo_que_no_existe_da_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        cargar_config_etapa2(tmp_path / "no_existe.yaml")
