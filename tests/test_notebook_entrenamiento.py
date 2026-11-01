"""Pruebas del notebook de entrenamiento (notebooks/02_entrenamiento_etapa1.ipynb).

El notebook solo orquesta: sin salidas ni datos personales, y lanza los comandos del repositorio.
Como aquí no hay Colab ni GPU, además de comprobar su estructura se ejecutan sus celdas con Drive,
git, torch y los comandos simulados, para verificar qué órdenes lanzan y qué comprueban.
"""

import ast
import json
import os
import re
import sys
import types
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
RUTA_NOTEBOOK = RAIZ / "notebooks" / "02_entrenamiento_etapa1.ipynb"
REPO_URL = "https://github.com/diegocalvofm-ux/TFG-VisionArtificial.git"

# Textos que no pueden aparecer en el notebook (sin distinguir mayúsculas), con las dos formas de
# escribir la barra invertida: la del texto y la del JSON crudo.
PROHIBIDOS = (
    "C:\\Users", "C:\\\\Users", "C:/Users", "/Users/", "token", "secret", "password", "Bearer",
)  # fmt: skip

# Qué debe contener cada celda de código, en el orden pedido.
CLAVES_POR_CELDA = [
    "REPO_URL",  # a) parámetros
    "torch.cuda.is_available",  # b) GPU
    "drive.mount",  # c) Drive
    "git clone",  # d) repositorio
    "src.train.entorno",  # e) instalar sin tocar torch
    "importar_publicos",  # f) datos
    "MODO",  # g) elegir smoke o completo
    "src.train.entrenar",  # h) entrenar
    "copytree",  # i) copiar a Drive
]


def fuente(celda: dict) -> str:
    texto = celda.get("source", "")
    return texto if isinstance(texto, str) else "".join(texto)


def celdas_de_codigo(cuaderno: dict) -> list[dict]:
    return [c for c in cuaderno.get("cells", []) if c.get("cell_type") == "code"]


def salidas(cuaderno: dict) -> list[str]:
    problemas = []
    for posicion, celda in enumerate(cuaderno.get("cells", [])):
        if celda.get("cell_type") == "code":
            if "outputs" not in celda or "execution_count" not in celda:
                problemas.append(f"celda {posicion}: faltan las claves outputs o execution_count")
            if celda.get("outputs"):
                problemas.append(f"celda {posicion}: tiene salidas")
            if celda.get("execution_count") is not None:
                problemas.append(f"celda {posicion}: tiene execution_count")
        elif "outputs" in celda:
            problemas.append(f"celda {posicion}: una celda de texto lleva outputs")
    return problemas


def prohibidos(cuaderno: dict, crudo: str) -> list[str]:
    texto = (
        crudo + json.dumps(cuaderno, ensure_ascii=False)
        + "".join(fuente(c) for c in cuaderno["cells"])
    ).lower()  # fmt: skip
    return [f"contiene {p!r}" for p in PROHIBIDOS if p.lower() in texto]


# --- Estructura -------------------------------------------------------------------------


@pytest.fixture(scope="module")
def notebook() -> dict:
    if not RUTA_NOTEBOOK.is_file():
        pytest.fail(f"No existe el notebook: {RUTA_NOTEBOOK.relative_to(RAIZ).as_posix()}")
    return json.loads(RUTA_NOTEBOOK.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def codigo(notebook) -> list[str]:
    """El texto de cada celda de código, en orden."""
    return [fuente(c) for c in celdas_de_codigo(notebook)]


def test_el_notebook_es_nbformat_4_con_kernel_python3_y_pide_gpu(notebook):
    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["kernelspec"]["name"] == "python3"
    assert notebook["metadata"].get("accelerator") == "GPU"


def test_el_notebook_no_tiene_salidas_ni_execution_count(notebook):
    assert salidas(notebook) == []


def test_el_notebook_no_contiene_rutas_de_usuario_ni_credenciales(notebook):
    assert prohibidos(notebook, RUTA_NOTEBOOK.read_text(encoding="utf-8")) == []


def test_cada_celda_tiene_un_id_unico(notebook):
    ids = [c.get("id") for c in notebook["cells"]]

    assert all(ids) and len(set(ids)) == len(ids)


def test_el_notebook_pasa_la_validacion_oficial_de_nbformat(notebook):
    nbformat = pytest.importorskip("nbformat")

    nbformat.validate(nbformat.from_dict(notebook))


def test_hay_nueve_celdas_de_codigo_en_el_orden_pedido(codigo):
    assert len(codigo) == len(CLAVES_POR_CELDA)
    for celda, clave in zip(codigo, CLAVES_POR_CELDA, strict=True):
        assert clave in celda, clave


def test_cada_celda_de_codigo_es_python_valido(codigo):
    for posicion, celda in enumerate(codigo):
        ast.parse(celda, filename=f"celda_{posicion}")


def test_el_notebook_solo_orquesta_no_importa_src_ni_rfdetr(codigo):
    for celda in codigo:
        assert not re.search(r"^\s*(from|import)\s+(src|rfdetr)\b", celda, re.MULTILINE)


def test_no_se_usan_magics_de_ipython(codigo):
    for celda in codigo:
        assert not re.search(r"^\s*[!%]", celda, re.MULTILINE)


def test_ninguna_orden_de_pip_instala_torch_ni_torchvision(codigo):
    instalaciones = [x for c in codigo for x in c.splitlines() if "pip install" in x]

    assert instalaciones
    for linea in instalaciones:
        assert "torch" not in linea.lower(), linea


def test_el_pip_usa_constraints_y_requirements_train(codigo):
    (linea,) = [x for x in codigo[4].splitlines() if "pip install" in x]

    assert " -c " in linea and "-r requirements-train.txt" in linea


def test_el_notebook_no_copia_la_carpeta_de_trabajo_a_drive(codigo):
    assert "TRABAJO" not in codigo[8]


def test_el_texto_final_explica_como_registrar_y_que_el_dataset_no_se_copia(notebook):
    final = [fuente(c) for c in notebook["cells"] if c["cell_type"] == "markdown"][-1]

    assert "/registrar-experimento" in final and "REGISTRO:" in final
    assert "resumen.json" in final
    assert "trabajo" in final.lower()


# --- Ejecución de las celdas con Colab, git, torch y comandos simulados -----------------


def _correr(texto: str, espacio: dict) -> None:
    exec(compile(texto, "<celda>", "exec"), espacio)  # noqa: S102


@pytest.fixture
def espacio(codigo, tmp_path) -> dict:
    """El espacio de nombres tras la celda a), con las rutas apuntando a una carpeta temporal."""
    ns = {"__name__": "notebook_colab"}
    _correr(codigo[0], ns)
    ns.update(
        CARPETA_DRIVE=str(tmp_path / "drive"),
        ZIPS_DRIVE=str(tmp_path / "drive" / "data" / "raw"),
        REPO=str(tmp_path / "repo"),
        RUNS=str(tmp_path / "repo" / "runs"),
        TRABAJO=str(tmp_path / "trabajo"),
    )
    return ns


class Ejecutor:
    """Sustituye a ``ejecutar``: apunta cada orden y devuelve el código de salida que se le diga."""

    def __init__(self, espacio: dict, rama: bool = False):
        self.espacio, self.rama, self.llamadas = espacio, rama, []

    def __call__(self, comando, cwd=None, comprobar=True):
        self.llamadas.append((comando, cwd))
        if comando.startswith("git clone"):
            os.makedirs(self.espacio["REPO"], exist_ok=True)
        if "symbolic-ref" in comando:
            return 0 if self.rama else 128
        return 0

    @property
    def comandos(self) -> list[str]:
        return [c for c, _ in self.llamadas]


@pytest.fixture
def colab_falso(monkeypatch):
    """Un ``google.colab.drive`` de mentira que apunta dónde se monta Drive."""
    montajes = []
    colab = types.ModuleType("google.colab")
    colab.drive = types.SimpleNamespace(mount=lambda ruta, **kw: montajes.append(ruta))
    google = types.ModuleType("google")
    google.colab = colab
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    return montajes


def _torch_falso(monkeypatch, *, cuda: bool) -> None:
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(
        is_available=lambda: cuda, get_device_name=lambda indice: "GPU de prueba X1"
    )
    monkeypatch.setitem(sys.modules, "torch", torch)


def test_a_los_parametros_tienen_los_valores_pedidos(codigo):
    ns = {}
    _correr(codigo[0], ns)

    assert ns["REPO_URL"] == REPO_URL and ns["COMMIT"] == "main"
    assert ns["CARPETA_DRIVE"] == "/content/drive/MyDrive/tfg-lechugas"
    assert ns["ZIPS_DRIVE"] == "/content/drive/MyDrive/tfg-lechugas/data/raw"
    assert ns["CONFIG"] == "configs/train/etapa1_rfdetr_n.yaml"
    assert ns["REPO"] == "/content/TFG-VisionArtificial"
    assert ns["RUNS"] == "/content/TFG-VisionArtificial/runs"
    assert ns["TRABAJO"] == "/content/trabajo"  # la copia de trabajo, fuera de Drive y de runs
    assert ns["EXP_ID"]


def test_b_sin_gpu_para_con_un_error_claro(codigo, espacio, monkeypatch):
    _torch_falso(monkeypatch, cuda=False)

    with pytest.raises(RuntimeError, match="No hay GPU"):
        _correr(codigo[1], espacio)


def test_b_con_gpu_imprime_su_nombre(codigo, espacio, monkeypatch, capsys):
    _torch_falso(monkeypatch, cuda=True)

    _correr(codigo[1], espacio)

    assert "GPU de prueba X1" in capsys.readouterr().out


def test_c_monta_drive_y_cuenta_los_zip(codigo, espacio, colab_falso, capsys):
    carpeta = Path(espacio["ZIPS_DRIVE"])
    carpeta.mkdir(parents=True)
    (carpeta / "a.zip").write_bytes(b"x")
    (carpeta / "B.ZIP").write_bytes(b"x")
    (carpeta / "notas.txt").write_text("no es un zip", encoding="utf-8")

    _correr(codigo[2], espacio)

    assert colab_falso == ["/content/drive"]
    assert "2 ZIP" in capsys.readouterr().out


def test_c_para_si_no_existe_la_carpeta_de_datos(codigo, espacio, colab_falso):
    with pytest.raises(FileNotFoundError, match="Sube los ZIP") as error:
        _correr(codigo[2], espacio)

    assert espacio["ZIPS_DRIVE"] in str(error.value)


def test_c_para_si_la_carpeta_no_tiene_ningun_zip(codigo, espacio, colab_falso):
    carpeta = Path(espacio["ZIPS_DRIVE"])
    carpeta.mkdir(parents=True)
    (carpeta / "notas.txt").write_text("hola", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="No hay ZIP"):
        _correr(codigo[2], espacio)


def test_d_clona_hace_checkout_y_entra_en_el_repo(codigo, espacio, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    ejecutor = espacio["ejecutar"] = Ejecutor(espacio)

    _correr(codigo[3], espacio)

    assert ejecutor.comandos[0] == f"git clone {REPO_URL} {espacio['REPO']}"
    assert "git checkout main" in ejecutor.comandos
    assert ejecutor.comandos[-1] == "git rev-parse --short HEAD"
    assert os.path.samefile(os.getcwd(), espacio["REPO"])


def test_d_si_el_repo_existe_hace_fetch_y_con_una_rama_la_pone_al_dia(
    codigo, espacio, monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    (Path(espacio["REPO"]) / ".git").mkdir(parents=True)
    ejecutor = espacio["ejecutar"] = Ejecutor(espacio, rama=True)

    _correr(codigo[3], espacio)

    assert not any(c.startswith("git clone") for c in ejecutor.comandos)
    assert "git fetch --all --tags" in ejecutor.comandos
    assert "git pull --ff-only" in ejecutor.comandos


def test_e_instala_sin_tocar_torch_y_comprueba_antes_y_despues(codigo, espacio):
    ejecutor = espacio["ejecutar"] = Ejecutor(espacio)

    _correr(codigo[4], espacio)

    comandos = ejecutor.comandos
    ordenes = [c.split("src.train.entorno ")[1].split()[0] if "src.train.entorno" in c else
               ("pip" if "pip install" in c else "?") for c in comandos]  # fmt: skip
    assert ordenes == ["versiones", "constraints", "pip", "versiones", "comprobar"]
    pip = comandos[2]
    assert "-c " in pip and "-r requirements-train.txt" in pip and "-r requirements.txt" in pip
    assert "torch" not in pip.lower()
    assert all(cwd == espacio["REPO"] for _, cwd in ejecutor.llamadas)
    # "antes" se guarda antes de instalar y "después" después; "comprobar" las compara
    assert "--out" in comandos[0] and "--out" in comandos[3]
    antes = comandos[0].split("--out ")[1]
    despues = comandos[3].split("--out ")[1]
    assert antes != despues
    assert f"--antes {antes}" in comandos[4] and f"--despues {despues}" in comandos[4]


def test_f_copia_solo_los_zip_a_data_raw_del_repo_y_lanza_el_importador(codigo, espacio):
    drive = Path(espacio["ZIPS_DRIVE"])
    drive.mkdir(parents=True)
    (drive / "svodk.zip").write_bytes(b"1")
    (drive / "giovi.zip").write_bytes(b"2")
    (drive / "notas.txt").write_text("no", encoding="utf-8")
    ejecutor = espacio["ejecutar"] = Ejecutor(espacio)

    _correr(codigo[5], espacio)

    destino = Path(espacio["REPO"]) / "data" / "raw"
    assert sorted(p.name for p in destino.iterdir()) == ["giovi.zip", "svodk.zip"]
    assert (destino / "svodk.zip").read_bytes() == b"1"
    assert (drive / "svodk.zip").exists()  # el original de Drive no se toca
    (comando,) = ejecutor.comandos
    assert comando.endswith(
        "-m src.data.importar_publicos importar --etapa configs/datasets/etapa1_v1.yaml"
    )
    assert ejecutor.llamadas[0][1] == espacio["REPO"]


def test_g_el_modo_por_defecto_es_smoke(codigo, espacio):
    _correr(codigo[6], espacio)

    assert espacio["MODO"] == "smoke"


def test_h_en_smoke_pasa_la_opcion_smoke_y_la_carpeta_de_trabajo(codigo, espacio):
    ejecutor = espacio["ejecutar"] = Ejecutor(espacio)
    espacio["MODO"] = "smoke"

    _correr(codigo[7], espacio)

    (comando,) = ejecutor.comandos
    assert "-m src.train.entrenar" in comando and comando.endswith(" --smoke")
    assert f"--config {espacio['CONFIG']}" in comando
    assert f"--exp-id {espacio['EXP_ID']}" in comando
    assert f"--trabajo {espacio['TRABAJO']}" in comando
    assert ejecutor.llamadas[0][1] == espacio["REPO"]


def test_h_en_modo_completo_no_pasa_smoke(codigo, espacio):
    ejecutor = espacio["ejecutar"] = Ejecutor(espacio)
    espacio["MODO"] = "completo"

    _correr(codigo[7], espacio)

    (comando,) = ejecutor.comandos
    assert "--smoke" not in comando and "--exp-id" in comando


def test_h_un_modo_desconocido_para_sin_entrenar(codigo, espacio):
    ejecutor = espacio["ejecutar"] = Ejecutor(espacio)
    espacio["MODO"] = "rapido"

    with pytest.raises(ValueError, match="smoke"):
        _correr(codigo[7], espacio)

    assert ejecutor.comandos == []


def test_i_copia_runs_exp_id_a_drive_y_no_copia_la_carpeta_de_trabajo(codigo, espacio, capsys):
    ejecucion = Path(espacio["RUNS"]) / espacio["EXP_ID"]
    (ejecucion / "pesos").mkdir(parents=True)
    (ejecucion / "resumen.json").write_text("{}", encoding="utf-8")
    (ejecucion / "pesos" / "checkpoint_best_total.pth").write_bytes(b"p" * 2048)
    trabajo = Path(espacio["TRABAJO"]) / espacio["EXP_ID"] / "train"
    trabajo.mkdir(parents=True)
    (trabajo / "svodk_0001.jpg").write_bytes(b"imagen")

    _correr(codigo[8], espacio)
    _correr(codigo[8], espacio)  # repetirla sobrescribe, no falla

    destino = Path(espacio["CARPETA_DRIVE"]) / "runs" / espacio["EXP_ID"]
    assert (destino / "resumen.json").is_file()
    assert (destino / "pesos" / "checkpoint_best_total.pth").stat().st_size == 2048
    assert not list(Path(espacio["CARPETA_DRIVE"]).rglob("*.jpg"))  # nada del dataset en Drive
    assert "checkpoint_best_total.pth" in capsys.readouterr().out
