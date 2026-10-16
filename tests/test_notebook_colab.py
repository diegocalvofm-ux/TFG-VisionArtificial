"""Pruebas del notebook de Colab (notebooks/01_v0_colab.ipynb).

El notebook solo orquesta: no tiene salidas ni datos personales, y lanza el pipeline del repo.
Como aquí no hay Colab ni GPU, además de comprobar su estructura se ejecutan sus celdas con
Drive, git y los comandos simulados, para verificar qué órdenes lanzan y qué comprueban.
"""

import ast
import json
import os
import re
import shlex
import subprocess
import sys
import types
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
RUTA_NOTEBOOK = RAIZ / "notebooks" / "01_v0_colab.ipynb"
REPO_URL = "https://github.com/diegocalvofm-ux/TFG-VisionArtificial.git"

# Textos que no pueden aparecer en el notebook (se comparan sin distinguir mayúsculas). Se
# incluyen las dos formas de escribir la barra invertida: la del texto y la del JSON crudo.
PROHIBIDOS = (
    "C:\\Users",
    "C:\\\\Users",
    "C:/Users",
    "/Users/",
    "token",
    "secret",
    "password",
    "Bearer",
)
OBLIGATORIOS_EN_EL_CODIGO = ("requirements.txt", "--detector owlvit")

# La celda e) para el notebook si no hay GPU y deja una línea lista para copiar al registrar.
MENSAJE_SIN_GPU = "No hay GPU: Entorno de ejecución -> Cambiar tipo de entorno -> GPU T4"
PREFIJO_REGISTRO = "REGISTRO:"


# --- Comprobadores: cada uno devuelve la lista de problemas que encuentra ------------------


def fuente(celda: dict) -> str:
    texto = celda.get("source", "")
    return texto if isinstance(texto, str) else "".join(texto)


def celdas_de_codigo(cuaderno: dict) -> list[dict]:
    return [c for c in cuaderno.get("cells", []) if c.get("cell_type") == "code"]


def formato(cuaderno: dict) -> list[str]:
    problemas = []
    if cuaderno.get("nbformat") != 4:
        problemas.append(f"nbformat debe ser 4 (es {cuaderno.get('nbformat')!r})")
    nucleo = cuaderno.get("metadata", {}).get("kernelspec", {})
    if nucleo.get("name") != "python3":
        problemas.append(f"el kernel debe ser python3 (es {nucleo.get('name')!r})")
    return problemas


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
            problemas.append(f"celda {posicion}: una celda que no es de código lleva outputs")
    return problemas


def prohibidos(cuaderno: dict, crudo: str = "") -> list[str]:
    """Textos prohibidos en el JSON (con las barras escapadas) y en el texto de las celdas."""
    texto = (crudo + json.dumps(cuaderno, ensure_ascii=False) + "".join(
        fuente(c) for c in cuaderno.get("cells", [])
    )).lower()
    return [f"contiene {p!r}" for p in PROHIBIDOS if p.lower() in texto]


def menciones(cuaderno: dict) -> list[str]:
    codigo = "\n".join(fuente(c) for c in celdas_de_codigo(cuaderno))
    return [f"el código no menciona {m!r}" for m in OBLIGATORIOS_EN_EL_CODIGO if m not in codigo]


def problemas(cuaderno: dict, crudo: str = "") -> list[str]:
    return formato(cuaderno) + salidas(cuaderno) + prohibidos(cuaderno, crudo) + menciones(cuaderno)


# --- Notebooks de ejemplo (para probar los comprobadores) -----------------------------------


def _codigo(texto: str, salidas_=None, cuenta=None) -> dict:
    return {
        "cell_type": "code",
        "execution_count": cuenta,
        "id": "c",
        "metadata": {},
        "outputs": salidas_ or [],
        "source": texto,
    }


def _cuaderno(*celdas: dict, nbformat: int = 4, kernel: str = "python3") -> dict:
    return {
        "cells": list(celdas),
        "metadata": {"kernelspec": {"name": kernel, "display_name": "Python 3"}},
        "nbformat": nbformat,
        "nbformat_minor": 5,
    }


LIMPIO = _cuaderno(
    {"cell_type": "markdown", "id": "m", "metadata": {}, "source": "# Título"},
    _codigo("pip install -r requirements.txt\npython -m src.pipeline.run --detector owlvit"),
)


def test_el_comprobador_da_por_bueno_un_notebook_limpio():
    assert problemas(LIMPIO) == []


def test_el_comprobador_detecta_un_notebook_de_ejemplo_con_salidas_y_execution_count():
    ejecutado = _cuaderno(
        _codigo(
            "pip install -r requirements.txt\npython -m src.pipeline.run --detector owlvit",
            salidas_=[{"output_type": "stream", "name": "stdout", "text": ["Procesadas: 4\n"]}],
            cuenta=3,
        )
    )

    encontrados = salidas(ejecutado)

    assert any("tiene salidas" in p for p in encontrados)
    assert any("tiene execution_count" in p for p in encontrados)
    assert len(problemas(ejecutado)) == 2  # solo eso falla: lo demás del ejemplo es correcto


def test_el_comprobador_detecta_un_execution_count_aunque_no_haya_salidas():
    assert any("execution_count" in p for p in salidas(_cuaderno(_codigo("x = 1", cuenta=1))))


def test_el_comprobador_detecta_las_claves_que_faltan_en_una_celda_de_codigo():
    sin_claves = _cuaderno({"cell_type": "code", "id": "c", "metadata": {}, "source": "x = 1"})

    assert any("faltan las claves" in p for p in salidas(sin_claves))


def test_el_comprobador_detecta_salidas_en_una_celda_de_texto():
    con_salida = _cuaderno({"cell_type": "markdown", "id": "m", "metadata": {}, "outputs": [],
                            "source": "texto"})  # fmt: skip

    assert any("no es de código" in p for p in salidas(con_salida))


@pytest.mark.parametrize("texto", ["C:\\Users\\ana\\x", "C:/Users/ana/x", "/Users/ana/x",
                                   "mi token", "un SECRET", "Password1", "Bearer abc"])  # fmt: skip
@pytest.mark.parametrize("donde", ["codigo", "texto"])
def test_el_comprobador_detecta_cada_texto_prohibido(texto, donde):
    celda = (
        _codigo(f"x = {texto!r}")
        if donde == "codigo"
        else {"cell_type": "markdown", "id": "m", "metadata": {}, "source": texto}
    )
    cuaderno = _cuaderno(LIMPIO["cells"][1], celda)

    assert prohibidos(cuaderno), texto


def test_el_comprobador_detecta_la_barra_invertida_tambien_en_el_json_crudo():
    crudo = '"source": "C:\\\\Users\\\\ana"'  # así queda escrito en un .ipynb

    assert prohibidos(LIMPIO, crudo)


@pytest.mark.parametrize(
    ("cuaderno", "fragmento"),
    [
        (_cuaderno(*LIMPIO["cells"], nbformat=3), "nbformat"),
        (_cuaderno(*LIMPIO["cells"], kernel="ir"), "kernel"),
        (_cuaderno(_codigo("python -m src.pipeline.run --detector owlvit")), "requirements.txt"),
        (_cuaderno(_codigo("pip install -r requirements.txt")), "--detector owlvit"),
    ],
    ids=["nbformat_3", "otro_kernel", "sin_requirements", "sin_detector_owlvit"],
)
def test_el_comprobador_detecta_cada_formato_o_mencion_incorrectos(cuaderno, fragmento):
    assert any(fragmento in p for p in problemas(cuaderno))


# --- El notebook real -----------------------------------------------------------------------


@pytest.fixture(scope="module")
def notebook() -> dict:
    if not RUTA_NOTEBOOK.is_file():
        pytest.fail(f"No existe el notebook: {RUTA_NOTEBOOK.relative_to(RAIZ).as_posix()}")
    return json.loads(RUTA_NOTEBOOK.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def codigo(notebook) -> list[str]:
    """El texto de cada celda de código, en orden."""
    return [fuente(c) for c in celdas_de_codigo(notebook)]


def test_el_notebook_es_nbformat_4_con_kernel_python3(notebook):
    assert formato(notebook) == []


def test_el_notebook_no_tiene_salidas_ni_execution_count(notebook):
    assert salidas(notebook) == []


def test_el_notebook_no_contiene_rutas_de_usuario_ni_credenciales(notebook):
    crudo = RUTA_NOTEBOOK.read_text(encoding="utf-8")

    assert prohibidos(notebook, crudo) == []


def test_las_celdas_de_codigo_mencionan_requirements_txt_y_detector_owlvit(notebook):
    assert menciones(notebook) == []


def test_el_notebook_cumple_todo_a_la_vez(notebook):
    assert problemas(notebook, RUTA_NOTEBOOK.read_text(encoding="utf-8")) == []


def test_el_notebook_pide_gpu_en_sus_metadatos(notebook):
    assert notebook["metadata"].get("accelerator") == "GPU"


def test_cada_celda_tiene_un_id_unico(notebook):
    ids = [c.get("id") for c in notebook["cells"]]

    assert all(ids) and len(set(ids)) == len(ids)


def test_el_notebook_pasa_la_validacion_oficial_de_nbformat(notebook):
    nbformat = pytest.importorskip("nbformat")

    nbformat.validate(nbformat.from_dict(notebook))  # lanza ValidationError si no es válido


def test_hay_ocho_celdas_de_codigo_en_el_orden_pedido(codigo):
    assert len(codigo) == 8
    claves = ["REPO_URL", "drive.mount", "git clone", "pip install", "torch.__version__",
              "_descarga_owlvit", "v0_colab_owlvit", "copytree"]  # fmt: skip
    for celda, clave in zip(codigo, claves, strict=True):
        assert clave in celda, clave


def test_cada_celda_de_codigo_es_python_valido(codigo):
    for posicion, celda in enumerate(codigo):
        ast.parse(celda, filename=f"celda_{posicion}")  # lanza SyntaxError si no lo es


def test_el_notebook_solo_orquesta_no_importa_src(codigo):
    for celda in codigo:
        assert not re.search(r"^\s*(from|import)\s+src\b", celda, re.MULTILINE)


def test_no_se_usan_magics_de_ipython(codigo):
    for celda in codigo:
        assert not re.search(r"^\s*[!%]", celda, re.MULTILINE)


def test_ninguna_orden_de_pip_instala_torch_ni_torchvision(codigo):
    instalaciones = [x for c in codigo for x in c.splitlines() if "pip install" in x]

    assert instalaciones  # hay al menos una
    for linea in instalaciones:
        assert "torch" not in linea.lower(), linea


def test_el_pip_instala_los_dos_requirements(codigo):
    celda = codigo[3]

    assert "-r requirements.txt" in celda and "-r requirements-detector.txt" in celda


def test_la_ultima_celda_de_texto_explica_como_registrar_el_resultado(notebook):
    textos = [fuente(c) for c in notebook["cells"] if c["cell_type"] == "markdown"]

    assert "/registrar-experimento exp001 runs/v0_colab_owlvit" in textos[-1]


def test_el_texto_final_pide_pegar_la_linea_de_registro_en_las_observaciones(notebook):
    final = [fuente(c) for c in notebook["cells"] if c["cell_type"] == "markdown"][-1]

    assert "/registrar-experimento exp001 runs/v0_colab_owlvit" in final
    assert PREFIJO_REGISTRO in final
    assert "observaciones" in final.lower()
    # el porqué: el JSON solo trae device=cuda; sin esa línea se pierden la GPU y transformers
    assert "device=cuda" in final
    assert "modelo de GPU" in final and "transformers" in final


# --- Ejecución de las celdas con Colab, git y comandos simulados ----------------------------


def _correr(texto: str, espacio: dict) -> None:
    exec(compile(texto, "<celda>", "exec"), espacio)  # noqa: S102


@pytest.fixture
def espacio(codigo, tmp_path) -> dict:
    """El espacio de nombres tras la celda a), con las rutas apuntando a una carpeta temporal."""
    ns = {"__name__": "notebook_colab"}
    _correr(codigo[0], ns)
    ns.update(
        CARPETA_DRIVE=str(tmp_path / "drive"),
        IMAGENES=str(tmp_path / "drive" / "fotos"),
        REPO=str(tmp_path / "repo"),
        RUNS=str(tmp_path / "runs"),
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
            return 0 if self.rama else 128  # 0 = está en una rama; 128 = HEAD desacoplado
        return 0

    @property
    def comandos(self) -> list[str]:
        return [c for c, _ in self.llamadas]


@pytest.fixture
def colab_falso(monkeypatch):
    """Un ``google.colab.drive`` de mentira que apunta dónde se monta Drive."""
    montajes = []
    drive = types.SimpleNamespace(mount=lambda ruta, **kw: montajes.append(ruta))
    colab = types.ModuleType("google.colab")
    colab.drive = drive
    google = types.ModuleType("google")
    google.colab = colab
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    return montajes


def _instalar_torch_falso(monkeypatch, *, cuda: bool) -> None:
    """Pone en su sitio un ``torch`` y un ``transformers`` de mentira, con versiones inventadas
    solo para la prueba, y una GPU que existe o no según ``cuda``."""
    torch = types.ModuleType("torch")
    torch.__version__ = "9.9.9+cuTEST"
    torch.cuda = types.SimpleNamespace(
        is_available=lambda: cuda, get_device_name=lambda indice: "GPU de prueba X1"
    )
    transformers = types.ModuleType("transformers")
    transformers.__version__ = "8.8.8"
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)


def test_a_los_parametros_tienen_los_valores_pedidos(espacio, codigo):
    ns = {}
    _correr(codigo[0], ns)  # sin las rutas de prueba

    assert ns["REPO_URL"] == REPO_URL
    assert ns["COMMIT"] == "main"
    assert ns["CARPETA_DRIVE"] == "/content/drive/MyDrive/tfg-lechugas"
    assert ns["IMAGENES"] == "/content/drive/MyDrive/tfg-lechugas/data/v0_test"


def test_a_ejecutar_muestra_la_salida_y_devuelve_el_codigo(espacio, capsys):
    # "ab" * 3 solo aparece como "ababab" en la salida, no en la línea del comando que se repite
    orden = shlex.join([sys.executable, "-c", "print('ab' * 3)"])

    codigo_salida = espacio["ejecutar"](orden)

    assert codigo_salida == 0 and "ababab" in capsys.readouterr().out


def test_a_ejecutar_para_con_un_error_si_el_comando_falla(espacio):
    orden = shlex.join([sys.executable, "-c", "import sys; sys.exit(3)"])

    with pytest.raises(RuntimeError, match="código 3"):
        espacio["ejecutar"](orden)


def test_a_ejecutar_puede_devolver_el_codigo_sin_parar(espacio):
    orden = shlex.join([sys.executable, "-c", "import sys; sys.exit(3)"])

    assert espacio["ejecutar"](orden, comprobar=False) == 3


def test_b_monta_drive_y_acepta_una_carpeta_con_fotos(codigo, espacio, colab_falso, capsys):
    carpeta = Path(espacio["IMAGENES"])
    carpeta.mkdir(parents=True)
    (carpeta / "FOTO.JPG").write_bytes(b"x")  # la extensión no distingue mayúsculas
    (carpeta / "notas.txt").write_text("no es una foto", encoding="utf-8")

    _correr(codigo[1], espacio)

    assert colab_falso == ["/content/drive"]
    assert "1 fotos" in capsys.readouterr().out


@pytest.mark.parametrize("extension", [".jpg", ".jpeg", ".png", ".webp", ".bmp"])
def test_b_admite_las_cinco_extensiones(codigo, espacio, colab_falso, extension):
    carpeta = Path(espacio["IMAGENES"])
    carpeta.mkdir(parents=True)
    (carpeta / f"foto{extension}").write_bytes(b"x")

    _correr(codigo[1], espacio)  # no lanza


def test_b_para_con_un_mensaje_claro_si_la_carpeta_no_existe(codigo, espacio, colab_falso):
    with pytest.raises(FileNotFoundError, match="Sube las fotos") as error:
        _correr(codigo[1], espacio)

    assert espacio["IMAGENES"] in str(error.value)


def test_b_para_si_la_carpeta_no_tiene_ninguna_imagen(codigo, espacio, colab_falso):
    carpeta = Path(espacio["IMAGENES"])
    carpeta.mkdir(parents=True)
    (carpeta / "notas.txt").write_text("hola", encoding="utf-8")
    (carpeta / "parece_foto.jpg").mkdir()  # una carpeta con nombre de foto no es una foto

    with pytest.raises(FileNotFoundError, match="No hay fotos"):
        _correr(codigo[1], espacio)


def test_c_si_el_repo_no_existe_lo_clona_hace_checkout_y_entra(codigo, espacio, monkeypatch):
    monkeypatch.chdir(espacio["RUNS"].rsplit(os.sep, 1)[0])
    ejecutor = espacio["ejecutar"] = Ejecutor(espacio)

    _correr(codigo[2], espacio)

    repo = espacio["REPO"]
    assert ejecutor.comandos[0] == f"git clone {REPO_URL} {repo}"
    assert "git checkout main" in ejecutor.comandos
    assert ejecutor.comandos[-1] == "git rev-parse --short HEAD"
    assert os.path.samefile(os.getcwd(), repo)  # entró en la carpeta del repo
    assert all("git fetch" not in c for c in ejecutor.comandos)


def test_c_si_el_repo_ya_existe_hace_fetch_en_vez_de_clonar(codigo, espacio, monkeypatch):
    Path(espacio["REPO"], ".git").mkdir(parents=True)
    ejecutor = espacio["ejecutar"] = Ejecutor(espacio)
    monkeypatch.chdir(Path(espacio["REPO"]).parent)

    _correr(codigo[2], espacio)

    assert ejecutor.comandos[0] == "git fetch --all --tags"
    assert ejecutor.llamadas[0][1] == espacio["REPO"]
    assert all("git clone" not in c for c in ejecutor.comandos)


def test_c_con_un_hash_hace_checkout_a_ese_commit_y_no_actualiza(codigo, espacio, monkeypatch):
    espacio["COMMIT"] = "a319672"
    ejecutor = espacio["ejecutar"] = Ejecutor(espacio, rama=False)  # HEAD desacoplado
    monkeypatch.chdir(Path(espacio["REPO"]).parent)

    _correr(codigo[2], espacio)

    assert "git checkout a319672" in ejecutor.comandos
    assert all("git pull" not in c for c in ejecutor.comandos)


def test_c_con_una_rama_la_pone_al_dia_despues_del_checkout(codigo, espacio, monkeypatch):
    ejecutor = espacio["ejecutar"] = Ejecutor(espacio, rama=True)
    monkeypatch.chdir(Path(espacio["REPO"]).parent)

    _correr(codigo[2], espacio)

    comandos = ejecutor.comandos
    assert comandos.index("git pull --ff-only") > comandos.index("git checkout main")


def test_d_instala_los_dos_requirements_desde_la_raiz_del_repo_y_sin_torch(codigo, espacio):
    ejecutor = espacio["ejecutar"] = Ejecutor(espacio)

    _correr(codigo[3], espacio)

    assert len(ejecutor.llamadas) == 1
    comando, cwd = ejecutor.llamadas[0]
    pedido = comando.split("pip install", 1)[1]
    assert "-r requirements.txt" in pedido and "-r requirements-detector.txt" in pedido
    assert "torch" not in pedido.lower()
    assert cwd == espacio["REPO"]


def test_e_imprime_versiones_gpu_y_transformers(codigo):
    celda = codigo[4]

    for esperado in ("sys.version", "torch.__version__", "torch.cuda.is_available()",
                     "get_device_name", "transformers.__version__"):  # fmt: skip
        assert esperado in celda, esperado


def test_e_contiene_la_comprobacion_de_gpu_y_el_runtimeerror_con_el_mensaje_pedido(codigo):
    celda = codigo[4]

    assert "torch.cuda.is_available()" in celda
    assert "RuntimeError" in celda
    assert MENSAJE_SIN_GPU in celda


def test_e_sin_gpu_para_el_notebook_con_runtimeerror_antes_de_imprimir_nada(
    codigo, espacio, monkeypatch, capsys
):
    _instalar_torch_falso(monkeypatch, cuda=False)

    with pytest.raises(RuntimeError) as error:
        _correr(codigo[4], espacio)

    assert str(error.value) == MENSAJE_SIN_GPU
    assert capsys.readouterr().out == ""  # la parada va antes de cualquier impresión


def test_e_con_gpu_imprime_una_sola_linea_de_registro_con_los_cinco_datos(
    codigo, espacio, monkeypatch, capsys
):
    _instalar_torch_falso(monkeypatch, cuda=True)
    pedidos = []

    def commit_falso(comando, **opciones):
        pedidos.append((" ".join(comando), opciones.get("cwd")))
        return "abc1234\n"  # como lo devuelve git: con salto de línea

    monkeypatch.setattr(subprocess, "check_output", commit_falso)

    _correr(codigo[4], espacio)

    lineas = [x for x in capsys.readouterr().out.splitlines() if x.startswith(PREFIJO_REGISTRO)]
    assert lineas == [
        f"{PREFIJO_REGISTRO} commit=abc1234 | GPU=GPU de prueba X1 | transformers=8.8.8"
        f" | Python={sys.version.split()[0]} | PyTorch=9.9.9+cuTEST"
    ]
    # el commit es el que dejó la celda c: el HEAD del repo clonado
    assert pedidos == [("git rev-parse --short HEAD", espacio["REPO"])]


def test_f_primera_ejecucion_descarga_el_modelo_con_una_repeticion_sin_calentamiento(
    codigo, espacio
):
    ejecutor = espacio["ejecutar"] = Ejecutor(espacio)

    _correr(codigo[5], espacio)

    comando, cwd = ejecutor.llamadas[0]
    assert len(ejecutor.llamadas) == 1 and cwd == espacio["REPO"]
    assert "-m src.pipeline.run" in comando
    assert "--detector owlvit" in comando
    assert "--repeticiones 1 --calentamiento 0" in comando
    assert f"--out {espacio['RUNS']}/_descarga_owlvit" in comando
    assert espacio["IMAGENES"] in comando


def test_g_la_medicion_valida_usa_los_valores_por_defecto_y_otra_carpeta(codigo, espacio):
    ejecutor = espacio["ejecutar"] = Ejecutor(espacio)

    _correr(codigo[6], espacio)

    comando, cwd = ejecutor.llamadas[0]
    assert len(ejecutor.llamadas) == 1 and cwd == espacio["REPO"]
    assert "-m src.pipeline.run" in comando and "--detector owlvit" in comando
    assert f"--out {espacio['RUNS']}/v0_colab_owlvit" in comando
    assert "_descarga_owlvit" not in comando
    assert "--repeticiones" not in comando and "--calentamiento" not in comando  # 5 y 1 por defecto


def test_h_copia_a_drive_solo_la_medicion_valida_y_lista_los_archivos(codigo, espacio, capsys):
    runs = Path(espacio["RUNS"])
    (runs / "v0_colab_owlvit").mkdir(parents=True)
    (runs / "v0_colab_owlvit" / "foto.json").write_text("{}", encoding="utf-8")
    (runs / "v0_colab_owlvit" / "foto_anotada.jpg").write_bytes(b"x" * 2048)
    (runs / "_descarga_owlvit").mkdir()
    (runs / "_descarga_owlvit" / "otra.json").write_text("{}", encoding="utf-8")

    _correr(codigo[7], espacio)

    destino = Path(espacio["CARPETA_DRIVE"]) / "runs"
    assert sorted(p.name for p in (destino / "v0_colab_owlvit").iterdir()) == [
        "foto.json",
        "foto_anotada.jpg",
    ]
    assert not (destino / "_descarga_owlvit").exists()  # la descarga no se copia
    salida = capsys.readouterr().out
    assert "foto.json" in salida and "foto_anotada.jpg" in salida


def test_h_se_puede_repetir_sin_error_y_sobrescribe(codigo, espacio):
    origen = Path(espacio["RUNS"]) / "v0_colab_owlvit"
    origen.mkdir(parents=True)
    (origen / "foto.json").write_text('{"v": 1}', encoding="utf-8")
    _correr(codigo[7], espacio)
    (origen / "foto.json").write_text('{"v": 2}', encoding="utf-8")

    _correr(codigo[7], espacio)  # dirs_exist_ok=True: no falla porque el destino ya existe

    copiado = Path(espacio["CARPETA_DRIVE"]) / "runs" / "v0_colab_owlvit" / "foto.json"
    assert copiado.read_text(encoding="utf-8") == '{"v": 2}'
