"""Pruebas de src/train/entorno.py: entorno de la ejecución y torch sin cambios."""

import json
from types import SimpleNamespace

from src.experimentos.privacidad import contiene_ruta_absoluta, contiene_usuario
from src.train import entorno
from src.train.entorno import comprobar_torch, entorno_entrenamiento, main

VERSIONES = {
    "torch": "2.11.0+cu130",
    "torchvision": "0.26.0+cu130",
    "rfdetr": "1.11.2",
    "transformers": "5.18.0",
    "pytorch-lightning": "2.6.6",
    "numpy": "2.0.2",
}


def _git(commit="abc1234", status="", falla=False):
    """Imita ``subprocess.run`` para git: rev-parse da el commit y status el estado."""

    def ejecutar(comando, **kwargs):
        if falla:
            raise FileNotFoundError("git")
        salida = commit + "\n" if "rev-parse" in comando else status
        return SimpleNamespace(returncode=0, stdout=salida)

    return ejecutar


def _gpu():
    return {"nombre": "GPU de prueba X1", "memoria_total_mb": 15360}


# --- comprobar_torch --------------------------------------------------------------------


def test_si_torch_y_torchvision_no_cambian_se_dice_y_es_ok():
    antes = {"torch": "2.11.0+cu130", "torchvision": "0.26.0+cu130"}

    sin_cambios, lineas = comprobar_torch(antes, dict(antes))

    assert sin_cambios is True
    assert lineas == [
        "torch: 2.11.0+cu130 -> 2.11.0+cu130 (sin cambios)",
        "torchvision: 0.26.0+cu130 -> 0.26.0+cu130 (sin cambios)",
    ]


def test_si_torch_cambia_se_detecta_y_se_dice_cual():
    antes = {"torch": "2.11.0+cu130", "torchvision": "0.26.0+cu130"}
    despues = {"torch": "2.12.0", "torchvision": "0.26.0+cu130"}

    sin_cambios, lineas = comprobar_torch(antes, despues)

    assert sin_cambios is False
    assert "torch: 2.11.0+cu130 -> 2.12.0 (CAMBIÓ)" in lineas
    assert "torchvision: 0.26.0+cu130 -> 0.26.0+cu130 (sin cambios)" in lineas


def test_si_cambia_torchvision_tambien_se_detecta():
    antes = {"torch": "1.0", "torchvision": "2.0"}

    sin_cambios, lineas = comprobar_torch(antes, {"torch": "1.0", "torchvision": "3.0"})

    assert sin_cambios is False and "torchvision: 2.0 -> 3.0 (CAMBIÓ)" in lineas


def test_que_aparezca_o_desaparezca_torch_cuenta_como_cambio():
    assert comprobar_torch({"torch": None}, {"torch": "2.0"})[0] is False
    assert comprobar_torch({"torch": "2.0"}, {"torch": None})[0] is False
    assert comprobar_torch({"torch": None}, {"torch": None})[0] is True


# --- línea de comandos ------------------------------------------------------------------


def test_versiones_guarda_torch_y_torchvision_en_un_json(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(entorno, "versiones_paquetes", lambda nombres=None: dict(VERSIONES))
    destino = tmp_path / "antes.json"

    codigo = main(["versiones", "--out", str(destino)])

    assert codigo == 0
    guardado = json.loads(destino.read_text(encoding="utf-8"))
    assert guardado["torch"] == "2.11.0+cu130" and guardado["torchvision"] == "0.26.0+cu130"
    assert "2.11.0+cu130" in capsys.readouterr().out


def test_constraints_escribe_torch_y_torchvision_con_su_version_exacta(tmp_path, monkeypatch):
    monkeypatch.setattr(entorno, "versiones_paquetes", lambda nombres=None: dict(VERSIONES))
    destino = tmp_path / "constraints.txt"

    codigo = main(["constraints", "--out", str(destino)])

    assert codigo == 0
    assert destino.read_text(encoding="utf-8").splitlines() == [
        "torch==2.11.0+cu130",
        "torchvision==0.26.0+cu130",
    ]


def test_constraints_sin_torch_instalado_da_error(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(entorno, "versiones_paquetes", lambda nombres=None: {"torch": None})

    codigo = main(["constraints", "--out", str(tmp_path / "c.txt")])

    assert codigo == 1 and "torch" in capsys.readouterr().err
    assert not (tmp_path / "c.txt").exists()


def _guardar(tmp_path, nombre, datos):
    ruta = tmp_path / nombre
    ruta.write_text(json.dumps(datos), encoding="utf-8")
    return str(ruta)


def test_comprobar_sale_con_0_si_no_cambio_nada_y_lo_imprime(tmp_path, capsys):
    datos = {"torch": "2.11.0", "torchvision": "0.26.0"}
    antes, despues = _guardar(tmp_path, "a.json", datos), _guardar(tmp_path, "d.json", datos)

    codigo = main(["comprobar", "--antes", antes, "--despues", despues])

    assert codigo == 0 and "(sin cambios)" in capsys.readouterr().out


def test_comprobar_sale_con_1_si_torch_cambio(tmp_path, capsys):
    antes = _guardar(tmp_path, "a.json", {"torch": "2.11.0", "torchvision": "0.26.0"})
    despues = _guardar(tmp_path, "d.json", {"torch": "2.12.0", "torchvision": "0.26.0"})

    codigo = main(["comprobar", "--antes", antes, "--despues", despues])

    salida = capsys.readouterr()
    assert codigo == 1 and "CAMBIÓ" in salida.out
    assert "torch" in salida.err


# --- entorno_entrenamiento --------------------------------------------------------------


def test_el_entorno_trae_versiones_gpu_commit_y_estado_de_git(tmp_path):
    env = entorno_entrenamiento(
        raiz=tmp_path, ejecutar=_git(), versiones=lambda: dict(VERSIONES), gpu=_gpu,
        plataforma="colab",
    )  # fmt: skip

    assert env["python"].count(".") == 2
    assert env["torch"] == "2.11.0+cu130" and env["torchvision"] == "0.26.0+cu130"
    assert env["rfdetr"] == "1.11.2" and env["transformers"] == "5.18.0"
    assert env["pytorch_lightning"] == "2.6.6" and env["numpy"] == "2.0.2"
    assert env["gpu"] == "GPU de prueba X1" and env["gpu_memoria_mb"] == 15360
    assert env["commit"] == "abc1234" and env["dirty"] is False
    assert env["plataforma"] == "colab"


def test_con_cambios_sin_commitear_dirty_es_true(tmp_path):
    env = entorno_entrenamiento(
        raiz=tmp_path, ejecutar=_git(status=" M src/x.py\n"), versiones=lambda: dict(VERSIONES),
        gpu=_gpu, plataforma="local",
    )  # fmt: skip

    assert env["dirty"] is True


def test_sin_git_el_commit_y_dirty_son_none_sin_error(tmp_path):
    env = entorno_entrenamiento(
        raiz=tmp_path, ejecutar=_git(falla=True), versiones=lambda: dict(VERSIONES), gpu=_gpu,
        plataforma="local",
    )  # fmt: skip

    assert env["commit"] is None and env["dirty"] is None


def test_sin_gpu_el_nombre_y_la_memoria_son_none(tmp_path):
    env = entorno_entrenamiento(
        raiz=tmp_path, ejecutar=_git(), versiones=lambda: dict(VERSIONES),
        gpu=lambda: {"nombre": None, "memoria_total_mb": None}, plataforma="local",
    )  # fmt: skip

    assert env["gpu"] is None and env["gpu_memoria_mb"] is None


def test_un_paquete_no_instalado_sale_como_none(tmp_path):
    versiones = dict(VERSIONES, rfdetr=None)

    env = entorno_entrenamiento(
        raiz=tmp_path, ejecutar=_git(), versiones=lambda: versiones, gpu=_gpu, plataforma="local"
    )

    assert env["rfdetr"] is None


def test_el_entorno_no_lleva_rutas_absolutas_ni_el_usuario_del_sistema(tmp_path):
    versiones = dict(VERSIONES, transformers="C:\\Users\\alguien\\transformers")

    env = entorno_entrenamiento(
        raiz=tmp_path, ejecutar=_git(), versiones=lambda: versiones, gpu=_gpu, plataforma="local"
    )

    texto = json.dumps(env)
    assert env["transformers"] is None  # la red de seguridad descarta el valor con ruta
    assert not contiene_ruta_absoluta(texto) and not contiene_usuario(texto)


def test_versiones_paquetes_lee_las_versiones_instaladas_sin_importar_torch():
    # pytest está instalado en el entorno de pruebas; un paquete inexistente da None.
    versiones = entorno.versiones_paquetes(("pytest", "paquete-que-no-existe-xyz"))

    assert versiones["pytest"] and versiones["paquete-que-no-existe-xyz"] is None
