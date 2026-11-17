"""Pruebas de src/train/etapa2/entrenar.py: entrenamiento de principio a fin en CPU.

Se entrena de verdad (2 épocas, imágenes de 32 px) pero con el extractor diminuto del conftest y
el dataset sintético: sin timm, sin descargar pesos y sin GPU. Tarda menos de un segundo.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from src.data.schema import GROUPS  # noqa: E402
from src.experimentos.privacidad import contiene_ruta_absoluta, contiene_usuario  # noqa: E402
from src.train.etapa2.config import cargar_config_etapa2  # noqa: E402
from src.train.etapa2.entrenar import (  # noqa: E402
    METRICA_PRINCIPAL,
    ejecutar,
    factor_lr,
    main,
    pasos_calentamiento,
    resolver_dispositivo,
)
from src.train.etapa2.modelo import cargar_modelo, predecir  # noqa: E402

ENTORNO = {
    "python": "3.13.15",
    "torch": "2.11.0+cu130",
    "torchvision": "0.26.0+cu130",
    "rfdetr": None,
    "transformers": None,
    "pytorch_lightning": None,
    "numpy": "2.0.2",
    "gpu": "GPU de prueba X1",
    "gpu_memoria_mb": 15360,
    "sistema": "Linux 6.1",
    "plataforma": "colab",
    "commit": "abc1234",
    "dirty": False,
    "timm": "1.0.30",
}
ARCHIVOS = ("config_usada.yaml", "entorno.json", "dataset_informe.json", "datos.json",
            "metricas_por_epoca.json", "matrices_confusion.json", "resumen.json")  # fmt: skip


def _leer(ruta: Path):
    texto = ruta.read_text(encoding="utf-8")
    return yaml.safe_load(texto) if ruta.suffix == ".yaml" else json.loads(texto)


def _correr(raiz: Path, ruta_config: Path, crear_extractor, *, exp_id="exp_a", smoke=False,
            **kw):  # fmt: skip
    config = cargar_config_etapa2(ruta_config)
    opciones = {"entorno": lambda: dict(ENTORNO), "memoria_gpu": lambda: None} | kw
    return ejecutar(config, exp_id, smoke=smoke, raiz=raiz, crear_extractor=crear_extractor,
                    **opciones)  # fmt: skip


@pytest.fixture
def proyecto(tmp_path, dataset_sintetico):
    """Una "raíz" falsa con el dataset en data/processed/etapa2_v1."""
    dataset_sintetico(tmp_path / "data" / "processed" / "etapa2_v1")
    return tmp_path


@pytest.fixture(scope="module")
def ejecucion(tmp_path_factory, fabricas):
    """Una ejecución completa (2 épocas) compartida por las pruebas que solo leen sus salidas."""
    raiz = tmp_path_factory.mktemp("proyecto")
    fabricas.escribir_dataset(raiz / "data" / "processed" / "etapa2_v1")
    ruta = fabricas.escribir_config(raiz, {"datos.sobremuestreo_propio": 3})
    resumen = _correr(raiz, ruta, fabricas.extractor)
    return raiz / "runs" / "exp_a", resumen


# --- programación de la tasa --------------------------------------------------------------


def test_factor_lr_sube_en_linea_recta_y_baja_en_coseno():
    factores = [factor_lr(paso, 11, 2, 0.1) for paso in range(11)]

    assert factores[0] == pytest.approx(0.5) and factores[1] == pytest.approx(1.0)
    assert factores[2] == pytest.approx(1.0)  # empieza el coseno
    assert factores[6] == pytest.approx(0.1 + 0.9 * 0.5)  # a mitad del coseno
    assert factores[10] == pytest.approx(0.1)  # el último paso llega al factor final
    assert all(a >= b for a, b in zip(factores[2:], factores[3:], strict=False))


def test_factor_lr_sin_calentamiento_empieza_en_1():
    assert factor_lr(0, 5, 0, 0.0) == pytest.approx(1.0)
    assert factor_lr(4, 5, 0, 0.0) == pytest.approx(0.0)
    assert factor_lr(2, 5, 0, 0.0) == pytest.approx(0.5 * (1 + math.cos(math.pi * 0.5)))


def test_factor_lr_con_un_solo_paso_es_1():
    assert factor_lr(0, 1, 0, 0.01) == pytest.approx(1.0)


def test_el_calentamiento_se_limita_a_la_mitad_de_los_pasos():
    assert pasos_calentamiento(1, 7, 70) == 7
    assert pasos_calentamiento(1, 7, 7) == 3  # smoke de una época: no se calienta todo
    assert pasos_calentamiento(0, 7, 70) == 0


def test_dispositivo_auto_sin_gpu_es_cpu_y_cuda_sin_gpu_da_error(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert resolver_dispositivo("auto") == "cpu" and resolver_dispositivo("cpu") == "cpu"
    with pytest.raises(RuntimeError, match="GPU"):
        resolver_dispositivo("cuda")


# --- archivos de la ejecución -------------------------------------------------------------


def test_se_escriben_todos_los_archivos_de_la_ejecucion(ejecucion):
    carpeta, _ = ejecucion

    for nombre in ARCHIVOS:
        assert (carpeta / nombre).is_file(), nombre
    assert sorted(p.name for p in (carpeta / "pesos").iterdir()) == ["mejor.pt", "ultimo.pt"]
    assert not list(carpeta.rglob("*.jpg"))  # ninguna imagen del dataset va a runs/


def test_metricas_por_epoca_tiene_perdidas_criterio_tasas_y_tiempo(ejecucion):
    carpeta, _ = ejecucion

    epocas = _leer(carpeta / "metricas_por_epoca.json")["epocas"]

    assert [e["epoca"] for e in epocas] == [1, 2]
    for epoca in epocas:
        assert epoca["perdida_train"] > 0 and epoca["valid"]["perdida"] > 0
        assert 0 <= epoca["valid"]["criterio"] <= 1
        assert set(epoca["valid"]["grupos"]) == set(GROUPS)
        assert len(epoca["lr"]) == 2 and epoca["tiempo_s"] >= 0
        assert epoca["pasos"] == math.ceil(52 / 4)  # 16 públicas + 12 propias x 3, lotes de 4


def test_el_mejor_punto_es_el_de_mayor_criterio_en_valid(ejecucion):
    carpeta, resumen = ejecucion
    epocas = _leer(carpeta / "metricas_por_epoca.json")["epocas"]

    criterios = [e["valid"]["criterio"] for e in epocas]
    esperada = criterios.index(max(criterios)) + 1  # en un empate, la primera

    assert resumen["mejor"]["epoca"] == esperada
    assert resumen["mejor"]["criterio"] == pytest.approx(max(criterios))


def test_valid_y_test_se_evaluan_con_el_mejor_punto(ejecucion):
    _, resumen = ejecucion

    assert resumen["valid"]["criterio"] == pytest.approx(resumen["mejor"]["criterio"])
    assert resumen["test"]["grupos"]["daño_bio"]["modelo"]["n"] == 8
    assert resumen["test"]["grupos"]["corte"]["modelo"]["n"] == 0
    assert resumen["valid"]["grupos_con_datos"] == ["daño_bio"]


def test_el_resumen_lleva_la_linea_base_mayoritaria_y_el_desglose_por_fuente(ejecucion):
    _, resumen = ejecucion

    mayoritaria = resumen["test"]["grupos"]["daño_bio"]["mayoritaria"]
    assert mayoritaria["clase"] == "BIO_OK"  # train: 18 BIO_OK, 7 DISEASE_VISIBLE, 3 ROT_DECAY
    assert mayoritaria["exactitud_equilibrada"] is not None
    assert list(resumen["test"]["por_fuente"]) == ["yssp", "phs"]  # propio no tiene test


def test_las_matrices_de_confusion_van_aparte(ejecucion):
    carpeta, resumen = ejecucion

    matrices = _leer(carpeta / "matrices_confusion.json")

    bio = matrices["test"]["grupos"]["daño_bio"]
    assert bio["etiquetas"] == list(GROUPS["daño_bio"])
    assert len(bio["matriz"]) == 5 and sum(map(sum, bio["matriz"])) == 8
    assert sum(map(sum, matrices["valid"]["por_fuente"]["phs"]["daño_bio"]["matriz"])) == 4
    assert "matriz" not in json.dumps(resumen, ensure_ascii=False)


def test_datos_json_cuenta_el_sobremuestreo_los_pesos_y_las_clases_ausentes(ejecucion):
    carpeta, _ = ejecucion

    datos = _leer(carpeta / "datos.json")

    assert datos["muestras_por_epoca"] == 16 + 12 * 3 and datos["sobremuestreo_propio"] == 3
    assert datos["recuentos"]["train"]["propio"]["recortes"] == 12  # sin las repeticiones
    assert set(datos["pesos_clase"]) == set(GROUPS)
    assert list(datos["pesos_clase"]["corte"]) == list(GROUPS["corte"])
    assert datos["pesos_clase"]["corte"]["UNCLEAN_CUT"] == 0  # sin muestras en train
    assert "VISIBLE_PEST" in datos["clases_sin_muestras_train"]["daño_bio"]
    assert datos["clases_mayoritarias"]["daño_bio"] == "BIO_OK"
    assert datos["filas_sin_etiquetas"] == [] and datos["reparto_propio"] is None


def test_el_resumen_trae_tiempo_pesos_registro_y_avisos(ejecucion):
    carpeta, resumen = ejecucion

    assert _leer(carpeta / "resumen.json") == resumen
    assert resumen["error"] is None and resumen["smoke"] is False
    assert resumen["epocas_pedidas"] == 2 and resumen["epocas_completadas"] == 2
    assert resumen["tiempo_entrenamiento_s"] >= 0 and resumen["memoria_gpu_max_mb"] is None
    assert resumen["dispositivo"] == "cpu" and resumen["precision_mixta"] is False
    assert [p["archivo"] for p in resumen["pesos"]] == ["pesos/mejor.pt", "pesos/ultimo.pt"]
    assert resumen["registro"] == {
        "commit": "abc1234", "entorno": "colab-gpu", "python": "3.13.15",
        "torch": "2.11.0+cu130", "metrica_principal": METRICA_PRINCIPAL,
        "valor": resumen["mejor"]["criterio"],
    }  # fmt: skip
    assert any("corte" in aviso and "valid" in aviso for aviso in resumen["avisos"])


def test_la_configuracion_usada_y_el_entorno_se_guardan(ejecucion):
    carpeta, _ = ejecucion

    usada = _leer(carpeta / "config_usada.yaml")

    assert usada["ejecucion"] == {"exp_id": "exp_a", "smoke": False}
    assert usada["datos"]["sobremuestreo_propio"] == 3 and usada["entrenamiento"]["epocas"] == 2
    assert usada["aumentos"]["proporcion_recorte"] == [1.0, 1.0]
    assert _leer(carpeta / "entorno.json") == ENTORNO
    assert _leer(carpeta / "dataset_informe.json")["semilla"] == 42


def test_ningun_archivo_de_la_ejecucion_lleva_rutas_absolutas_ni_el_usuario(ejecucion):
    carpeta, _ = ejecucion

    for ruta in carpeta.rglob("*"):
        if ruta.suffix in (".json", ".yaml"):
            texto = ruta.read_text(encoding="utf-8")
        elif ruta.suffix == ".pt":
            datos = torch.load(ruta, map_location="cpu", weights_only=True)
            texto = str({k: v for k, v in datos.items() if k != "estado"})
        else:
            continue
        assert not contiene_ruta_absoluta(texto), ruta.name
        assert not contiene_usuario(texto), ruta.name


def test_los_pesos_guardados_se_cargan_y_predicen(ejecucion, extractor_diminuto):
    carpeta, resumen = ejecucion

    ruta = carpeta / "pesos" / "mejor.pt"
    modelo = cargar_modelo(ruta, "cpu", crear_extractor=extractor_diminuto)
    datos = torch.load(ruta, map_location="cpu", weights_only=True)

    assert datos["epoca"] == resumen["mejor"]["epoca"]
    (resultado,) = predecir(modelo, [np.zeros((16, 24, 3), dtype=np.uint8)])
    assert set(resultado) == set(GROUPS)


def test_se_imprime_la_linea_registro_y_el_comando_para_registrar(proyecto, escribir_config,
                                                                  extractor_diminuto, capsys):
    resumen = _correr(proyecto, escribir_config(), extractor_diminuto)

    salida = capsys.readouterr().out
    assert ("REGISTRO: commit=abc1234 | GPU=GPU de prueba X1 | timm=1.0.30 | Python=3.13.15 | "
            "PyTorch=2.11.0+cu130") in salida  # fmt: skip
    assert (
        "python -m src.experimentos.registro campos --exp expNNN --set commit=abc1234 "
        "--set entorno=colab-gpu --set python=3.13.15 --set torch=2.11.0+cu130 "
        f"--set metrica_principal={METRICA_PRINCIPAL} --set valor={resumen['mejor']['criterio']}"
    ) in salida
    assert "época 1/2" in salida and "época 2/2" in salida


def test_con_arbol_sucio_se_avisa(proyecto, escribir_config, extractor_diminuto, capsys):
    sucio = ENTORNO | {"dirty": True}

    _correr(proyecto, escribir_config(), extractor_diminuto, entorno=lambda: dict(sucio))

    assert "cambios sin commitear" in capsys.readouterr().out


def test_la_misma_semilla_da_las_mismas_perdidas(proyecto, escribir_config, extractor_diminuto):
    ruta = escribir_config()
    _correr(proyecto, ruta, extractor_diminuto, exp_id="exp_a")
    _correr(proyecto, ruta, extractor_diminuto, exp_id="exp_b")

    a = _leer(proyecto / "runs" / "exp_a" / "metricas_por_epoca.json")["epocas"]
    b = _leer(proyecto / "runs" / "exp_b" / "metricas_por_epoca.json")["epocas"]
    assert [e["perdida_train"] for e in a] == [e["perdida_train"] for e in b]


# --- smoke y reparto ----------------------------------------------------------------------


def test_smoke_usa_una_epoca_y_los_limites_del_yaml(proyecto, escribir_config, extractor_diminuto,
                                                    capsys):
    resumen = _correr(proyecto, escribir_config(), extractor_diminuto, smoke=True)

    carpeta = proyecto / "runs" / "exp_a"
    datos = _leer(carpeta / "datos.json")
    totales = {p: sum(f["recortes"] for f in por_fuente.values())
               for p, por_fuente in datos["recuentos"].items()}  # fmt: skip
    assert totales == {"train": 6, "valid": 4, "test": 4}  # smoke.recortes_train y recortes_eval
    assert "propio" in datos["recuentos"]["train"]  # el smoke ejercita también la fuente propia
    assert resumen["smoke"] is True and resumen["epocas_pedidas"] == 1
    assert len(_leer(carpeta / "metricas_por_epoca.json")["epocas"]) == 1
    usada = _leer(carpeta / "config_usada.yaml")
    assert usada["ejecucion"]["smoke"] is True and usada["entrenamiento"]["epocas"] == 1
    salida = capsys.readouterr().out
    assert "REGISTRO:" in salida and "registro campos" not in salida and "smoke" in salida


def test_el_reparto_propio_lleva_lechugas_propias_a_valid_y_test(proyecto, escribir_config,
                                                                 extractor_diminuto):
    reparto = {"train": 0.5, "valid": 0.25, "test": 0.25}

    resumen = _correr(proyecto, escribir_config({"datos.reparto_propio": reparto}),
                      extractor_diminuto)  # fmt: skip

    datos = _leer(proyecto / "runs" / "exp_a" / "datos.json")
    lechugas = datos["reparto_propio"]["lechugas"]
    assert sorted(sum(lechugas.values(), [])) == [f"L{i}" for i in range(1, 7)]
    assert datos["recuentos"]["valid"]["propio"]["recortes"] > 0
    assert "propio" in resumen["test"]["por_fuente"]


# --- errores ------------------------------------------------------------------------------


class _ExtractorQueFalla(nn.Module):
    def __init__(self, error: Exception):
        super().__init__()
        self.capa = nn.Linear(1, 1)
        self.error = error

    def forward(self, x):
        raise self.error


def test_si_el_entrenamiento_falla_se_guarda_el_error_sin_inventar_metricas(
    proyecto, escribir_config
):
    def fabrica(arquitectura, preentrenado):
        return _ExtractorQueFalla(RuntimeError("sin memoria en la GPU")), 4

    with pytest.raises(RuntimeError, match="sin memoria"):
        _correr(proyecto, escribir_config(), fabrica)

    carpeta = proyecto / "runs" / "exp_a"
    resumen = _leer(carpeta / "resumen.json")
    assert "sin memoria en la GPU" in resumen["error"]
    assert resumen["mejor"] is None and resumen["valid"] is None and resumen["test"] is None
    assert resumen["registro"]["valor"] is None and resumen["pesos"] == []
    assert (carpeta / "config_usada.yaml").is_file() and (carpeta / "entorno.json").is_file()


def test_el_error_guardado_no_lleva_rutas_absolutas(proyecto, escribir_config):
    ruta = proyecto / "algo" / "grande.bin"

    def fabrica(arquitectura, preentrenado):
        raise OSError(f"no se pudo descargar en {ruta}")

    with pytest.raises(OSError, match="no se pudo descargar"):
        _correr(proyecto, escribir_config(), fabrica)

    texto = (proyecto / "runs" / "exp_a" / "resumen.json").read_text(encoding="utf-8")
    assert not contiene_ruta_absoluta(texto) and not contiene_usuario(texto)


@pytest.mark.parametrize("exp_id", ["../fuera", "con espacio", "", "a/b", ".oculto"])
def test_un_exp_id_no_valido_da_error(proyecto, escribir_config, extractor_diminuto, exp_id):
    with pytest.raises(ValueError, match="exp-id"):
        _correr(proyecto, escribir_config(), extractor_diminuto, exp_id=exp_id)

    assert not (proyecto / "runs").exists()


def test_una_ejecucion_que_ya_existe_no_se_pisa(proyecto, escribir_config, extractor_diminuto):
    carpeta = proyecto / "runs" / "exp_a"
    carpeta.mkdir(parents=True)
    (carpeta / "mis_pesos.pt").write_bytes(b"importantes")

    with pytest.raises(ValueError, match="ya existe"):
        _correr(proyecto, escribir_config(), extractor_diminuto)

    assert (carpeta / "mis_pesos.pt").read_bytes() == b"importantes"


def test_sin_el_dataset_se_explica_como_generarlo(tmp_path, escribir_config, extractor_diminuto):
    with pytest.raises(FileNotFoundError, match="src.data.etapa2"):
        _correr(tmp_path, escribir_config(), extractor_diminuto)

    assert not (tmp_path / "runs").exists()


def test_solo_se_escribe_en_runs_y_el_dataset_no_cambia(proyecto, escribir_config,
                                                        extractor_diminuto):
    ruta_config = escribir_config()
    antes = {p: p.read_bytes() for p in proyecto.rglob("*") if p.is_file()}

    _correr(proyecto, ruta_config, extractor_diminuto)

    nuevos = [p for p in proyecto.rglob("*") if p.is_file() and p not in antes]
    assert nuevos and all((proyecto / "runs" / "exp_a") in p.parents for p in nuevos)
    for ruta, contenido in antes.items():
        assert ruta.read_bytes() == contenido


# --- línea de comandos --------------------------------------------------------------------


def test_main_ejecuta_con_las_inyecciones(proyecto, escribir_config, extractor_diminuto, capsys):
    codigo = main(
        ["--config", str(escribir_config()), "--exp-id", "exp_cli", "--raiz", str(proyecto),
         "--smoke"],
        crear_extractor=extractor_diminuto, entorno=lambda: dict(ENTORNO),
        memoria_gpu=lambda: None,
    )  # fmt: skip

    assert codigo == 0
    assert (proyecto / "runs" / "exp_cli" / "resumen.json").is_file()
    assert "REGISTRO:" in capsys.readouterr().out


def test_main_con_un_yaml_no_valido_sale_con_1_y_nombra_la_clave(proyecto, escribir_config,
                                                                 capsys):
    ruta = escribir_config({"aumentos.brillo": 0.5})

    codigo = main(["--config", str(ruta), "--exp-id", "e", "--raiz", str(proyecto)])

    assert codigo == 1 and "aumentos.brillo" in capsys.readouterr().err


def test_main_oculta_las_rutas_absolutas_en_los_errores(proyecto, escribir_config, capsys):
    codigo = main(
        ["--config", str(escribir_config()), "--exp-id", "e", "--raiz", str(proyecto / "nada")]
    )

    error = capsys.readouterr().err
    assert codigo == 1 and "ERROR:" in error
    assert not contiene_ruta_absoluta(error) and not contiene_usuario(error)
