"""Pruebas de src/train/entrenar.py con el entrenamiento simulado (sin rfdetr ni GPU)."""

import json
import sys
import types
from pathlib import Path

import pytest

from src.experimentos.privacidad import contiene_ruta_absoluta, contiene_usuario
from src.train import entrenar
from src.train.config import cargar_config_entrenamiento
from src.train.entrenar import ejecutar, entrenar_rfdetr, main

ENTORNO = {
    "python": "3.13.15",
    "torch": "2.11.0+cu130",
    "torchvision": "0.26.0+cu130",
    "rfdetr": "1.11.2",
    "transformers": "5.18.0",
    "pytorch_lightning": "2.6.6",
    "numpy": "2.0.2",
    "gpu": "GPU de prueba X1",
    "gpu_memoria_mb": 15360,
    "sistema": "Linux 6.1",
    "plataforma": "colab",
    "commit": "abc1234",
    "dirty": False,
}

METRICS_CSV = """epoch,step,train/loss,val/mAP_50,val/mAP_50_95
0,9,0.9,,
0,19,,0.50,0.30
1,29,0.7,,
1,39,,0.70,0.45
2,49,0.5,,
2,59,,0.65,0.40
"""


class EntrenadorFalso:
    """Sustituye a ``entrenar_rfdetr``: apunta lo que recibe y escribe lo que escribiría rfdetr."""

    def __init__(self, csv=METRICS_CSV, falla=None, escribe_csv=True):
        self.csv, self.falla, self.escribe_csv = csv, falla, escribe_csv
        self.llamadas = []

    def __call__(self, parametros, dataset_dir, salida):
        self.llamadas.append((dict(parametros), Path(dataset_dir), Path(salida)))
        if self.falla:
            raise self.falla
        salida = Path(salida)
        salida.mkdir(parents=True, exist_ok=True)
        if self.escribe_csv:
            (salida / "metrics.csv").write_text(self.csv, encoding="utf-8")
        for nombre in ("checkpoint_best_total.pth", "checkpoint_best_ema.pth", "last.ckpt"):
            (salida / nombre).write_bytes(b"pesos falsos " + nombre.encode())
        (salida / "training_config.json").write_text("{}", encoding="utf-8")


class Reloj:
    """Un reloj falso: cada llamada avanza 12,5 s (así el tiempo medido es 12,5)."""

    def __init__(self):
        self.t = 0.0

    def __call__(self):
        self.t += 12.5
        return self.t


@pytest.fixture
def proyecto(tmp_path, dataset_sintetico):
    """Una "raíz" falsa con el dataset en data/processed/etapa1_v1, como tras el importador."""
    dataset_sintetico(tmp_path / "data" / "processed" / "etapa1_v1")
    return tmp_path


def _correr(proyecto, escribir_config, *, cambios=None, exp_id="exp_a", smoke=False, **kw):
    config = cargar_config_entrenamiento(escribir_config(cambios))
    entrenador = kw.pop("entrenador", None) or EntrenadorFalso()
    resumen = ejecutar(
        config, exp_id, smoke=smoke, raiz=proyecto, entrenador=entrenador,
        entorno=lambda: dict(ENTORNO), reloj=Reloj(), memoria_gpu=lambda: 1234.5, **kw,
    )  # fmt: skip
    return resumen, entrenador


def _coco(carpeta: Path) -> dict:
    return json.loads((carpeta / "_annotations.coco.json").read_text(encoding="utf-8"))


def _todos(carpeta: Path) -> list[Path]:
    return sorted(p for p in carpeta.rglob("*") if p.is_file())


# --- lo que recibe el entrenador --------------------------------------------------------


def test_el_entrenador_recibe_exactamente_lo_que_dice_el_yaml(proyecto, escribir_config):
    _, entrenador = _correr(proyecto, escribir_config)

    ((parametros, dataset_dir, salida),) = entrenador.llamadas
    assert parametros == {
        "clase": "RFDETRNano", "resolucion": 384, "clases": ["lechuga"], "epocas": 30,
        "batch_size": 8, "grad_accum_steps": 2, "lr": 1.0e-4, "lr_encoder": 1.5e-4,
        "amp_dtype": "auto", "num_workers": 2, "early_stopping": False,
        "checkpoint_interval": 10, "tensorboard": False, "dispositivo": "auto", "semilla": 42,
    }  # fmt: skip
    # entrena sobre la copia de trabajo (fuera de runs/) y escribe en runs/<exp_id>/rfdetr
    assert dataset_dir == proyecto / "data" / "processed" / "trabajo" / "exp_a"
    assert salida == proyecto / "runs" / "exp_a" / "rfdetr"


def test_los_cambios_del_yaml_llegan_al_entrenador(proyecto, escribir_config):
    cambios = {"entrenamiento.epocas": 7, "entrenamiento.lr": 5.0e-5, "semilla": 3,
               "modelo.resolucion": 416, "entrenamiento.batch_size": 4}  # fmt: skip

    _, entrenador = _correr(proyecto, escribir_config, cambios=cambios)

    parametros = entrenador.llamadas[0][0]
    assert (parametros["epocas"], parametros["lr"], parametros["semilla"]) == (7, 5.0e-5, 3)
    assert (parametros["resolucion"], parametros["batch_size"]) == (416, 4)


# --- archivos de la ejecución -----------------------------------------------------------


def test_se_escriben_todos_los_archivos_de_la_ejecucion(proyecto, escribir_config):
    _correr(proyecto, escribir_config)

    carpeta = proyecto / "runs" / "exp_a"
    for nombre in ("config_usada.yaml", "entorno.json", "dataset_informe.json",
                   "metricas_valid.json", "resumen.json"):  # fmt: skip
        assert (carpeta / nombre).is_file(), nombre
    assert (carpeta / "rfdetr" / "metrics.csv").is_file()
    assert (carpeta / "rfdetr" / "training_config.json").is_file()
    assert (carpeta / "rfdetr" / "last.ckpt").is_file()


def test_los_mejores_pesos_van_a_pesos_y_se_mueven_no_se_duplican(proyecto, escribir_config):
    _correr(proyecto, escribir_config)

    carpeta = proyecto / "runs" / "exp_a"
    assert sorted(p.name for p in (carpeta / "pesos").iterdir()) == [
        "checkpoint_best_ema.pth", "checkpoint_best_total.pth"
    ]  # fmt: skip
    assert not (carpeta / "rfdetr" / "checkpoint_best_total.pth").exists()
    assert not (carpeta / "rfdetr" / "checkpoint_best_ema.pth").exists()


def test_el_informe_del_dataset_se_copia_tal_cual(proyecto, escribir_config):
    _correr(proyecto, escribir_config)

    original = proyecto / "data" / "processed" / "etapa1_v1" / "informe.json"
    copia = proyecto / "runs" / "exp_a" / "dataset_informe.json"
    assert json.loads(copia.read_text(encoding="utf-8")) == json.loads(
        original.read_text(encoding="utf-8")
    )


def test_la_configuracion_usada_y_el_entorno_se_guardan(proyecto, escribir_config):
    import yaml

    _correr(proyecto, escribir_config)

    carpeta = proyecto / "runs" / "exp_a"
    usada = yaml.safe_load((carpeta / "config_usada.yaml").read_text(encoding="utf-8"))
    assert usada["ejecucion"] == {"exp_id": "exp_a", "smoke": False}
    assert usada["entrenamiento"]["epocas"] == 30 and usada["semilla"] == 42
    assert usada["datos"]["repeticiones_svodk"] == 1
    entorno = json.loads((carpeta / "entorno.json").read_text(encoding="utf-8"))
    assert entorno == ENTORNO


def test_las_metricas_de_valid_por_epoca_y_el_mejor_punto(proyecto, escribir_config):
    _correr(proyecto, escribir_config)

    guardado = json.loads(
        (proyecto / "runs" / "exp_a" / "metricas_valid.json").read_text(encoding="utf-8")
    )
    assert [m["epoca"] for m in guardado["metricas"]] == [1, 2, 3]
    assert [m["AP50"] for m in guardado["metricas"]] == [0.50, 0.70, 0.65]
    assert [m["AP50_95"] for m in guardado["metricas"]] == [0.30, 0.45, 0.40]
    assert guardado["mejor"] == {"epoca": 2, "epoca_csv": 1, "AP50": 0.70, "AP50_95": 0.45}


def test_el_resumen_trae_mejor_punto_tiempo_pesos_y_recuentos(proyecto, escribir_config):
    resumen, _ = _correr(proyecto, escribir_config)

    en_disco = json.loads(
        (proyecto / "runs" / "exp_a" / "resumen.json").read_text(encoding="utf-8")
    )
    assert en_disco == resumen
    assert resumen["exp_id"] == "exp_a" and resumen["smoke"] is False and resumen["error"] is None
    assert resumen["mejor"] == {"epoca": 2, "epoca_csv": 1, "AP50": 0.70, "AP50_95": 0.45}
    assert resumen["tiempo_entrenamiento_s"] == 12.5
    assert resumen["epocas_pedidas"] == 30 and resumen["epocas_con_metricas"] == 3
    assert resumen["memoria_gpu_max_mb"] == 1234.5
    assert [p["archivo"] for p in resumen["pesos"]] == [
        "pesos/checkpoint_best_ema.pth", "pesos/checkpoint_best_total.pth"
    ]  # fmt: skip
    assert resumen["recuentos"]["train"] == {
        "imagenes": 8, "anotaciones": 8, "imagenes_con_prefijo": 3
    }  # fmt: skip
    assert resumen["recuentos"]["valid"]["imagenes"] == 2
    assert resumen["recuentos"]["test"]["imagenes"] == 0


def test_el_resumen_lleva_los_campos_para_registrar_el_experimento(proyecto, escribir_config):
    resumen, _ = _correr(proyecto, escribir_config)

    assert resumen["registro"] == {
        "commit": "abc1234", "entorno": "colab-gpu", "python": "3.13.15",
        "torch": "2.11.0+cu130", "metrica_principal": "AP50_95", "valor": 0.45,
    }  # fmt: skip


def test_se_imprime_la_linea_registro_y_el_comando_para_registrar(
    proyecto, escribir_config, capsys
):
    _correr(proyecto, escribir_config)

    salida = capsys.readouterr().out
    assert (
        "REGISTRO: commit=abc1234 | GPU=GPU de prueba X1 | rfdetr=1.11.2 | "
        "Python=3.13.15 | PyTorch=2.11.0+cu130" in salida
    )
    assert (
        "python -m src.experimentos.registro campos --exp expNNN --set commit=abc1234 "
        "--set entorno=colab-gpu --set python=3.13.15 --set torch=2.11.0+cu130 "
        "--set metrica_principal=AP50_95 --set valor=0.45"
    ) in salida


def test_con_arbol_sucio_se_avisa(proyecto, escribir_config, capsys):
    config = cargar_config_entrenamiento(escribir_config())

    ejecutar(
        config, "exp_a", raiz=proyecto, entrenador=EntrenadorFalso(), reloj=Reloj(),
        entorno=lambda: dict(ENTORNO, dirty=True), memoria_gpu=lambda: None,
    )  # fmt: skip

    assert "cambios sin commitear" in capsys.readouterr().out


# --- smoke y repeticiones ---------------------------------------------------------------


def test_smoke_fuerza_las_epocas_del_yaml_y_el_subconjunto(proyecto, escribir_config):
    resumen, entrenador = _correr(proyecto, escribir_config, smoke=True)

    parametros, dataset_dir, _ = entrenador.llamadas[0]
    assert parametros["epocas"] == 1  # smoke.epocas, no las 30 del entrenamiento completo
    assert len(_coco(dataset_dir / "train")["images"]) == 4  # smoke.imagenes_train
    assert len(_coco(dataset_dir / "valid")["images"]) == 2  # todo valid
    assert resumen["smoke"] is True and resumen["epocas_pedidas"] == 1
    assert resumen["recuentos"]["train"]["imagenes"] == 4


def test_smoke_se_anota_en_la_configuracion_usada_y_no_propone_registrar(
    proyecto, escribir_config, capsys
):
    import yaml

    _correr(proyecto, escribir_config, smoke=True)

    usada = yaml.safe_load(
        (proyecto / "runs" / "exp_a" / "config_usada.yaml").read_text(encoding="utf-8")
    )
    assert usada["ejecucion"]["smoke"] is True and usada["entrenamiento"]["epocas"] == 1
    salida = capsys.readouterr().out
    assert "REGISTRO:" in salida
    assert "registro campos" not in salida and "smoke" in salida


def test_sin_smoke_se_usa_todo_train(proyecto, escribir_config):
    _, entrenador = _correr(proyecto, escribir_config)

    assert len(_coco(entrenador.llamadas[0][1] / "train")["images"]) == 8


def test_repeticiones_svodk_se_aplican_a_train_y_nunca_a_valid(proyecto, escribir_config):
    resumen, entrenador = _correr(
        proyecto, escribir_config, cambios={"datos.repeticiones_svodk": 3}
    )

    trabajo = entrenador.llamadas[0][1]
    train = [i["file_name"] for i in _coco(trabajo / "train")["images"]]
    valid = [i["file_name"] for i in _coco(trabajo / "valid")["images"]]
    assert sum(n.startswith("svodk_") for n in train) == 9
    assert sum(n.startswith("giovi_") for n in train) == 5
    assert len(valid) == 2 and not any("__r" in n for n in valid)
    assert resumen["recuentos"]["train"]["imagenes"] == 14
    assert resumen["recuentos"]["valid"]["imagenes"] == 2


# --- dónde se escribe -------------------------------------------------------------------


def test_el_dataset_no_se_escribe_en_la_carpeta_de_la_ejecucion(proyecto, escribir_config):
    _correr(proyecto, escribir_config, cambios={"datos.repeticiones_svodk": 2})

    carpeta = proyecto / "runs" / "exp_a"
    assert not [p for p in _todos(carpeta) if p.suffix in (".jpg", ".png")]
    assert not [p for p in _todos(carpeta) if p.name == "_annotations.coco.json"]
    assert (proyecto / "data" / "processed" / "trabajo" / "exp_a" / "train").is_dir()


def test_el_trabajo_se_puede_poner_en_otra_carpeta(proyecto, escribir_config, tmp_path):
    otra = tmp_path / "fuera" / "trabajo"

    _, entrenador = _correr(proyecto, escribir_config, trabajo=otra)

    assert entrenador.llamadas[0][1] == otra / "exp_a"
    assert (otra / "exp_a" / "train" / "_annotations.coco.json").is_file()
    assert not (proyecto / "data" / "processed" / "trabajo").exists()


def test_solo_se_escribe_en_runs_y_en_trabajo_y_el_dataset_original_no_cambia(
    proyecto, escribir_config
):
    antes = {p: p.read_bytes() for p in _todos(proyecto)}
    config_yaml = escribir_config()
    antes[config_yaml] = config_yaml.read_bytes()

    _correr(proyecto, escribir_config)

    nuevos = [p for p in _todos(proyecto) if p not in antes]
    permitidos = (
        proyecto / "runs" / "exp_a",
        proyecto / "data" / "processed" / "trabajo" / "exp_a",
    )
    assert nuevos and all(any(d in p.parents for d in permitidos) for p in nuevos)
    for ruta, contenido in antes.items():
        if ruta.exists():
            assert ruta.read_bytes() == contenido  # nada de lo que había se tocó


def test_ningun_archivo_de_la_ejecucion_lleva_rutas_absolutas_ni_el_usuario(
    proyecto, escribir_config
):
    _correr(proyecto, escribir_config, trabajo=proyecto / "afuera")

    carpeta = proyecto / "runs" / "exp_a"
    for ruta in _todos(carpeta):
        if ruta.suffix in (".json", ".yaml"):
            texto = ruta.read_text(encoding="utf-8")
            assert not contiene_ruta_absoluta(texto), ruta.name
            assert not contiene_usuario(texto), ruta.name


# --- errores ----------------------------------------------------------------------------


def test_si_el_entrenador_falla_se_guarda_el_error_sin_inventar_metricas(proyecto, escribir_config):
    entrenador = EntrenadorFalso(falla=RuntimeError("sin memoria en la GPU"))

    with pytest.raises(RuntimeError, match="sin memoria"):
        _correr(proyecto, escribir_config, entrenador=entrenador)

    carpeta = proyecto / "runs" / "exp_a"
    resumen = json.loads((carpeta / "resumen.json").read_text(encoding="utf-8"))
    assert "sin memoria en la GPU" in resumen["error"]
    assert resumen["mejor"] is None and resumen["pesos"] == []
    assert resumen["registro"]["valor"] is None
    assert not (carpeta / "metricas_valid.json").exists()
    assert (carpeta / "config_usada.yaml").is_file() and (carpeta / "entorno.json").is_file()


def test_el_error_guardado_no_lleva_rutas_absolutas(proyecto, escribir_config):
    ruta = proyecto / "algo" / "grande.bin"
    entrenador = EntrenadorFalso(falla=OSError(f"no se puede escribir {ruta}"))

    with pytest.raises(OSError, match="no se puede escribir"):
        _correr(proyecto, escribir_config, entrenador=entrenador)

    texto = (proyecto / "runs" / "exp_a" / "resumen.json").read_text(encoding="utf-8")
    assert not contiene_ruta_absoluta(texto) and not contiene_usuario(texto)


def test_si_no_hay_metrics_csv_el_error_se_guarda_y_no_se_inventa_nada(proyecto, escribir_config):
    entrenador = EntrenadorFalso(escribe_csv=False)

    with pytest.raises(FileNotFoundError, match="metrics.csv"):
        _correr(proyecto, escribir_config, entrenador=entrenador)

    resumen = json.loads((proyecto / "runs" / "exp_a" / "resumen.json").read_text(encoding="utf-8"))
    assert "metrics.csv" in resumen["error"]
    assert resumen["mejor"] is None and resumen["registro"]["valor"] is None


@pytest.mark.parametrize("exp_id", ["../fuera", "con espacio", "", "a/b", ".oculto"])
def test_un_exp_id_no_valido_da_error(proyecto, escribir_config, exp_id):
    with pytest.raises(ValueError, match="exp-id"):
        _correr(proyecto, escribir_config, exp_id=exp_id)

    assert not (proyecto / "runs").exists()


def test_una_ejecucion_que_ya_existe_no_se_pisa(proyecto, escribir_config):
    carpeta = proyecto / "runs" / "exp_a"
    carpeta.mkdir(parents=True)
    (carpeta / "mis_pesos.pth").write_bytes(b"importantes")

    with pytest.raises(ValueError, match="ya existe"):
        _correr(proyecto, escribir_config)

    assert (carpeta / "mis_pesos.pth").read_bytes() == b"importantes"


def test_sin_el_dataset_se_explica_como_generarlo(tmp_path, escribir_config):
    config = cargar_config_entrenamiento(escribir_config())

    with pytest.raises(FileNotFoundError, match="importar_publicos"):
        ejecutar(config, "exp_a", raiz=tmp_path, entrenador=EntrenadorFalso())

    assert not (tmp_path / "runs").exists()


# --- línea de comandos ------------------------------------------------------------------


def test_main_ejecuta_con_el_entrenador_inyectado(proyecto, escribir_config, capsys):
    ruta = escribir_config()
    entrenador = EntrenadorFalso()

    codigo = main(
        ["--config", str(ruta), "--exp-id", "exp_cli", "--raiz", str(proyecto), "--smoke"],
        entrenador=entrenador, entorno=lambda: dict(ENTORNO), reloj=Reloj(),
        memoria_gpu=lambda: None,
    )  # fmt: skip

    assert codigo == 0
    assert entrenador.llamadas[0][0]["epocas"] == 1
    assert (proyecto / "runs" / "exp_cli" / "resumen.json").is_file()
    assert "REGISTRO:" in capsys.readouterr().out


def test_main_pasa_la_carpeta_de_trabajo(proyecto, escribir_config, tmp_path):
    entrenador = EntrenadorFalso()

    main(
        ["--config", str(escribir_config()), "--exp-id", "e", "--raiz", str(proyecto),
         "--trabajo", str(tmp_path / "t")],
        entrenador=entrenador, entorno=lambda: dict(ENTORNO), reloj=Reloj(),
        memoria_gpu=lambda: None,
    )  # fmt: skip

    assert entrenador.llamadas[0][1] == tmp_path / "t" / "e"


def test_main_con_un_yaml_invalido_sale_con_1_y_el_mensaje(proyecto, escribir_config, capsys):
    ruta = escribir_config({"entrenamiento.epocas": 0})

    codigo = main(["--config", str(ruta), "--exp-id", "e", "--raiz", str(proyecto)])

    assert codigo == 1 and "entrenamiento.epocas" in capsys.readouterr().err


def test_main_oculta_las_rutas_absolutas_en_los_errores(proyecto, escribir_config, capsys):
    codigo = main(
        ["--config", str(escribir_config()), "--exp-id", "e", "--raiz", str(proyecto / "nada")]
    )

    error = capsys.readouterr().err
    assert codigo == 1 and "ERROR:" in error
    assert not contiene_ruta_absoluta(error) and not contiene_usuario(error)


# --- entrenar_rfdetr contra una librería falsa ------------------------------------------


class _RFDETRNanoFalso:
    instancias: list = []

    def __init__(self, **kwargs):
        self.kwargs_init = kwargs
        self.kwargs_train = None
        type(self).instancias.append(self)

    def train(self, **kwargs):
        self.kwargs_train = kwargs


@pytest.fixture
def rfdetr_falso(monkeypatch):
    """Un módulo ``rfdetr`` de mentira (y un ``torch`` sin GPU): no hace falta la librería."""
    _RFDETRNanoFalso.instancias = []
    modulo = types.ModuleType("rfdetr")
    modulo.RFDETRNano = _RFDETRNanoFalso
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(sys.modules, "rfdetr", modulo)
    monkeypatch.setitem(sys.modules, "torch", torch)
    return modulo


PARAMETROS = {
    "clase": "RFDETRNano", "resolucion": 384, "clases": ["lechuga"], "epocas": 30,
    "batch_size": 8, "grad_accum_steps": 2, "lr": 1.0e-4, "lr_encoder": 1.5e-4,
    "amp_dtype": "auto", "num_workers": 2, "early_stopping": False, "checkpoint_interval": 10,
    "tensorboard": False, "dispositivo": "auto", "semilla": 42,
}  # fmt: skip


def test_entrenar_rfdetr_llama_a_train_con_los_nombres_reales_de_la_libreria(
    rfdetr_falso, tmp_path
):
    entrenar_rfdetr(dict(PARAMETROS), tmp_path / "ds", tmp_path / "salida")

    (modelo,) = _RFDETRNanoFalso.instancias
    # Son los campos de rfdetr.config.TrainConfig 1.11.2 (extra="forbid": uno mal escrito falla).
    assert modelo.kwargs_train == {
        "dataset_dir": str(tmp_path / "ds"), "output_dir": str(tmp_path / "salida"),
        "epochs": 30, "batch_size": 8, "grad_accum_steps": 2, "lr": 1.0e-4,
        "lr_encoder": 1.5e-4, "resolution": 384, "seed": 42, "num_workers": 2,
        "amp_dtype": "auto", "early_stopping": False, "checkpoint_interval": 10,
        "tensorboard": False, "class_names": ["lechuga"], "device": "cpu",
    }  # fmt: skip


def test_con_dispositivo_cuda_o_auto_con_gpu_se_pide_cuda(rfdetr_falso, monkeypatch, tmp_path):
    entrenar_rfdetr(dict(PARAMETROS, dispositivo="cuda"), tmp_path, tmp_path / "s")
    sys.modules["torch"].cuda.is_available = lambda: True
    entrenar_rfdetr(dict(PARAMETROS, dispositivo="auto"), tmp_path, tmp_path / "s")
    entrenar_rfdetr(dict(PARAMETROS, dispositivo="cpu"), tmp_path, tmp_path / "s")

    dispositivos = [m.kwargs_train["device"] for m in _RFDETRNanoFalso.instancias]
    assert dispositivos == ["cuda", "cuda", "cpu"]


def test_entrenar_rfdetr_usa_la_clase_del_yaml_por_su_nombre(rfdetr_falso, tmp_path):
    class Otra(_RFDETRNanoFalso):
        pass

    rfdetr_falso.RFDETRSmall = Otra

    entrenar_rfdetr(dict(PARAMETROS, clase="RFDETRSmall"), tmp_path, tmp_path / "s")

    assert isinstance(_RFDETRNanoFalso.instancias[-1], Otra)


def test_si_rfdetr_no_esta_instalado_el_error_dice_como_instalarlo(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "rfdetr", None)  # import rfdetr -> ImportError

    with pytest.raises(ImportError, match="requirements-train.txt"):
        entrenar_rfdetr(dict(PARAMETROS), tmp_path, tmp_path / "s")


def test_el_modulo_no_importa_rfdetr_al_cargarse():
    codigo = Path(entrenar.__file__).read_text(encoding="utf-8")

    for linea in codigo.splitlines():
        if linea.startswith(("import rfdetr", "from rfdetr")):
            pytest.fail(f"importación de rfdetr al cargar el módulo: {linea}")
