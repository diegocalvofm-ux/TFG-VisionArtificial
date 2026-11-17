"""Pruebas del notebook de entrenamiento de la etapa 2 (notebooks/03_entrenamiento_etapa2.ipynb).

Mismo patrón que tests/test_notebook_entrenamiento.py: se comprueba la estructura (sin salidas ni
datos personales, sin instalar torch) y se ejecutan sus celdas con Drive, git, torch y los
comandos simulados para verificar qué órdenes lanzan y qué comprueban.
"""

import ast
import json
import os
import re
import sys
import types
import zipfile
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
RUTA_NOTEBOOK = RAIZ / "notebooks" / "03_entrenamiento_etapa2.ipynb"
REPO_URL = "https://github.com/diegocalvofm-ux/TFG-VisionArtificial.git"

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
    "zipfile",  # f) descomprimir los recortes
    "MODO",  # g) elegir smoke o completo
    "src.train.etapa2",  # h) entrenar
    "copytree",  # i) copiar a Drive
]


def fuente(celda: dict) -> str:
    texto = celda.get("source", "")
    return texto if isinstance(texto, str) else "".join(texto)


@pytest.fixture(scope="module")
def notebook() -> dict:
    if not RUTA_NOTEBOOK.is_file():
        pytest.fail(f"No existe el notebook: {RUTA_NOTEBOOK.relative_to(RAIZ).as_posix()}")
    return json.loads(RUTA_NOTEBOOK.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def codigo(notebook) -> list[str]:
    return [fuente(c) for c in notebook["cells"] if c.get("cell_type") == "code"]


# --- Estructura -------------------------------------------------------------------------


def test_el_notebook_es_nbformat_4_con_kernel_python3_y_pide_gpu(notebook):
    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["kernelspec"]["name"] == "python3"
    assert notebook["metadata"].get("accelerator") == "GPU"


def test_el_notebook_no_tiene_salidas_ni_execution_count(notebook):
    for posicion, celda in enumerate(notebook["cells"]):
        if celda["cell_type"] == "code":
            assert celda.get("outputs") == [], f"celda {posicion}: tiene salidas"
            assert "execution_count" in celda and celda["execution_count"] is None, posicion
        else:
            assert "outputs" not in celda, posicion


def test_el_notebook_no_contiene_rutas_de_usuario_ni_credenciales(notebook):
    crudo = RUTA_NOTEBOOK.read_text(encoding="utf-8")
    texto = (crudo + "".join(fuente(c) for c in notebook["cells"])).lower()

    assert [p for p in PROHIBIDOS if p.lower() in texto] == []


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


def test_el_notebook_solo_orquesta_no_importa_src_ni_timm(codigo):
    for celda in codigo:
        assert not re.search(r"^\s*(from|import)\s+(src|timm)\b", celda, re.MULTILINE)


def test_no_se_usan_magics_de_ipython(codigo):
    for celda in codigo:
        assert not re.search(r"^\s*[!%]", celda, re.MULTILINE)


def test_ninguna_orden_de_pip_instala_torch_ni_torchvision(codigo):
    instalaciones = [x for c in codigo for x in c.splitlines() if "pip install" in x]

    assert instalaciones
    for linea in instalaciones:
        assert "torch" not in linea.lower(), linea


def test_el_texto_final_explica_como_registrar(notebook):
    final = [fuente(c) for c in notebook["cells"] if c["cell_type"] == "markdown"][-1]

    assert "/registrar-experimento" in final and "REGISTRO:" in final
    assert "resumen.json" in final


# --- Ejecución de las celdas con Colab, git, torch y comandos simulados -----------------


def _correr(texto: str, espacio: dict) -> None:
    exec(compile(texto, "<celda>", "exec"), espacio)  # noqa: S102


@pytest.fixture
def espacio(codigo, tmp_path) -> dict:
    """El espacio de nombres tras la celda a), con las rutas apuntando a una carpeta temporal."""
    ns = {"__name__": "notebook_colab"}
    _correr(codigo[0], ns)
    repo = tmp_path / "repo"
    ns.update(
        CARPETA_DRIVE=str(tmp_path / "drive"),
        ZIP_DATOS=str(tmp_path / "drive" / "data" / "processed" / "etapa2_v1.zip"),
        REPO=str(repo),
        RUNS=str(repo / "runs"),
        DATASET=str(repo / "data" / "processed" / "etapa2_v1"),
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


def _zip(ruta: Path, prefijo: str = "", con_csv: bool = True) -> Path:
    """Un ZIP como el de data/processed/etapa2_v1, con o sin la carpeta delante."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ruta, "w") as zf:
        if con_csv:
            zf.writestr(f"{prefijo}recortes.csv", "recorte_id\nyssp_000001\n")
        zf.writestr(f"{prefijo}informe.json", "{}")
        zf.writestr(f"{prefijo}imagenes/yssp_000001.jpg", b"jpg")
        zf.writestr(f"{prefijo}imagenes/phs_000001.jpg", b"jpg")
    return ruta


def test_a_los_parametros_tienen_los_valores_pedidos(codigo):
    ns = {}
    _correr(codigo[0], ns)

    assert ns["REPO_URL"] == REPO_URL and ns["COMMIT"] == "main"
    assert ns["CARPETA_DRIVE"] == "/content/drive/MyDrive/tfg-lechugas"
    assert ns["ZIP_DATOS"] == "/content/drive/MyDrive/tfg-lechugas/data/processed/etapa2_v1.zip"
    assert ns["CONFIG"] == "configs/train/etapa2_effb0.yaml"
    assert ns["REPO"] == "/content/TFG-VisionArtificial"
    assert ns["RUNS"] == "/content/TFG-VisionArtificial/runs"
    # donde lo espera datos.dataset del YAML: en /content, no en Drive
    assert ns["DATASET"] == "/content/TFG-VisionArtificial/data/processed/etapa2_v1"
    assert ns["EXP_ID"]


def test_b_sin_gpu_para_con_un_error_claro(codigo, espacio, monkeypatch):
    _torch_falso(monkeypatch, cuda=False)

    with pytest.raises(RuntimeError, match="No hay GPU"):
        _correr(codigo[1], espacio)


def test_b_con_gpu_imprime_su_nombre(codigo, espacio, monkeypatch, capsys):
    _torch_falso(monkeypatch, cuda=True)

    _correr(codigo[1], espacio)

    assert "GPU de prueba X1" in capsys.readouterr().out


def test_c_monta_drive_y_comprueba_el_zip(codigo, espacio, colab_falso, capsys):
    _zip(Path(espacio["ZIP_DATOS"]))

    _correr(codigo[2], espacio)

    assert colab_falso == ["/content/drive"]
    assert "etapa2_v1.zip" in capsys.readouterr().out


def test_c_para_si_no_esta_el_zip(codigo, espacio, colab_falso):
    with pytest.raises(FileNotFoundError, match="etapa2_v1.zip") as error:
        _correr(codigo[2], espacio)

    assert espacio["ZIP_DATOS"] in str(error.value)


def test_d_clona_hace_checkout_y_entra_en_el_repo(codigo, espacio, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    ejecutor = espacio["ejecutar"] = Ejecutor(espacio)

    _correr(codigo[3], espacio)

    assert ejecutor.comandos[0] == f"git clone {REPO_URL} {espacio['REPO']}"
    assert "git checkout main" in ejecutor.comandos
    assert ejecutor.comandos[-1] == "git rev-parse --short HEAD"
    assert os.path.samefile(os.getcwd(), espacio["REPO"])


def test_e_instala_sin_tocar_torch_y_comprueba_antes_y_despues(codigo, espacio):
    ejecutor = espacio["ejecutar"] = Ejecutor(espacio)

    _correr(codigo[4], espacio)

    comandos = ejecutor.comandos
    ordenes = [c.split("src.train.entorno ")[1].split()[0] if "src.train.entorno" in c else
               ("pip" if "pip install" in c else "?") for c in comandos]  # fmt: skip
    assert ordenes == ["versiones", "constraints", "pip", "versiones", "comprobar"]
    pip = comandos[2]
    assert " -c " in pip and "-r requirements-train.txt" in pip and "-r requirements.txt" in pip
    assert "torch" not in pip.lower()
    assert all(cwd == espacio["REPO"] for _, cwd in ejecutor.llamadas)
    antes = comandos[0].split("--out ")[1]
    despues = comandos[3].split("--out ")[1]
    assert antes != despues
    assert f"--antes {antes}" in comandos[4] and f"--despues {despues}" in comandos[4]


@pytest.mark.parametrize("prefijo", ["", "etapa2_v1/"])
def test_f_descomprime_el_zip_donde_lo_espera_el_yaml(codigo, espacio, prefijo, capsys):
    _zip(Path(espacio["ZIP_DATOS"]), prefijo)

    _correr(codigo[5], espacio)

    dataset = Path(espacio["DATASET"])
    assert (dataset / "recortes.csv").is_file() and (dataset / "informe.json").is_file()
    assert sorted(p.name for p in (dataset / "imagenes").iterdir()) == [
        "phs_000001.jpg", "yssp_000001.jpg"
    ]  # fmt: skip
    assert not (dataset / "etapa2_v1").exists()  # nunca una carpeta de más
    assert "2 imágenes" in capsys.readouterr().out
    # en Drive solo sigue estando el ZIP
    drive = Path(espacio["CARPETA_DRIVE"])
    assert [p.name for p in drive.rglob("*") if p.is_file()] == ["etapa2_v1.zip"]


def test_f_un_zip_sin_recortes_csv_para_con_un_error_claro(codigo, espacio):
    _zip(Path(espacio["ZIP_DATOS"]), con_csv=False)

    with pytest.raises(ValueError, match="recortes.csv"):
        _correr(codigo[5], espacio)

    assert not Path(espacio["DATASET"]).exists()


def test_f_si_ya_esta_descomprimido_no_lo_repite(codigo, espacio, capsys):
    _zip(Path(espacio["ZIP_DATOS"]))
    _correr(codigo[5], espacio)
    marca = Path(espacio["DATASET"]) / "recortes.csv"
    marca.write_text("ya estaba", encoding="utf-8")

    _correr(codigo[5], espacio)

    assert marca.read_text(encoding="utf-8") == "ya estaba"
    assert "ya está descomprimido" in capsys.readouterr().out


def test_g_el_modo_por_defecto_es_smoke(codigo, espacio):
    _correr(codigo[6], espacio)

    assert espacio["MODO"] == "smoke"


def test_h_en_smoke_pasa_la_opcion_smoke(codigo, espacio):
    ejecutor = espacio["ejecutar"] = Ejecutor(espacio)
    espacio["MODO"] = "smoke"

    _correr(codigo[7], espacio)

    (comando,) = ejecutor.comandos
    assert "-m src.train.etapa2 " in comando and comando.endswith(" --smoke")
    assert f"--config {espacio['CONFIG']}" in comando
    assert f"--exp-id {espacio['EXP_ID']}" in comando
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


def test_i_copia_solo_runs_exp_id_a_drive(codigo, espacio, capsys):
    ejecucion = Path(espacio["RUNS"]) / espacio["EXP_ID"]
    (ejecucion / "pesos").mkdir(parents=True)
    (ejecucion / "resumen.json").write_text("{}", encoding="utf-8")
    (ejecucion / "pesos" / "mejor.pt").write_bytes(b"p" * 2048)
    dataset = Path(espacio["DATASET"]) / "imagenes"
    dataset.mkdir(parents=True)
    (dataset / "yssp_000001.jpg").write_bytes(b"imagen")

    _correr(codigo[8], espacio)
    _correr(codigo[8], espacio)  # repetirla sobrescribe, no falla

    destino = Path(espacio["CARPETA_DRIVE"]) / "runs" / espacio["EXP_ID"]
    assert (destino / "resumen.json").is_file()
    assert (destino / "pesos" / "mejor.pt").stat().st_size == 2048
    assert not list(Path(espacio["CARPETA_DRIVE"]).rglob("*.jpg"))  # nada del dataset en Drive
    assert "mejor.pt" in capsys.readouterr().out
