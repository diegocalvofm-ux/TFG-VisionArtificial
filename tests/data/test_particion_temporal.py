"""Pruebas del reparto temporal por fuente (``particion_temporal``): un solo vídeo."""

import random
import re
import shutil
from pathlib import Path

import pytest

from src.config import load_config
from src.data.publicos import (
    cargar_etapa,
    cargar_fuente,
    cargar_fuentes,
    construir_informe,
    exportar,
    filas_metadata,
    formatear_informe,
    importar,
    leer_fuente,
    repartir,
    tiempo_de,
)

RAIZ_REPO = Path(__file__).resolve().parents[2]
TABLA = load_config(RAIZ_REPO / "configs" / "salida_global.yaml")

PATRON = r"frame(\d+)-(\d+)-(\d+)-(\d+)_"
RECT = [[2, 2, 10, 2, 10, 8, 2, 8]]
CLASES = {"lechuga": "lechuga"}
FRACCIONES = {"train": 0.70, "valid": 0.15, "test": 0.15}


def _nombre(t: int) -> str:
    """Nombre de un fotograma al segundo ``t`` (formato de svodk: frameH-MM-SS-cc)."""
    return f"frame{t // 3600}-{(t % 3600) // 60:02d}-{t % 60:02d}-00_jpg.rf.x.jpg"


def _carpetas(n: int = 100, semilla_orden: int = 3) -> dict:
    """``n`` fotogramas (t = 0…n-1 s), con una lechuga cada uno y en orden BARAJADO en el ZIP."""
    nombres = [_nombre(t) for t in range(n)]
    random.Random(semilla_orden).shuffle(nombres)
    return {"train": [{"nombre": x, "anotaciones": [("lechuga", RECT)]} for x in nombres]}


def _temporal(**cambios) -> dict:
    return {"patron": PATRON, "fracciones": FRACCIONES, "margen_s": 5, **cambios}


def _video(fuente_sintetica, tmp_path, *, n=100, temporal=None, **extra):
    """Crea la fuente ``vid`` (reparto temporal) y devuelve (fuente, imágenes, particiones)."""
    ruta = fuente_sintetica(
        "vid",
        _carpetas(n),
        CLASES,
        papel="split",
        particion_temporal=_temporal() if temporal is None else temporal,
        **extra,
    )
    fuente = cargar_fuente(ruta, tmp_path)
    imagenes = leer_fuente(fuente)
    particiones, _ = repartir(imagenes, {"vid": fuente}, 42)
    return fuente, imagenes, particiones


def _tiempos(imagenes, particiones) -> dict[str, list[int]]:
    """Segundos de las imágenes de cada partición (ordenados)."""
    salida: dict[str, list[int]] = {"train": [], "valid": [], "test": []}
    for imagen in imagenes:
        if imagen.image_id in particiones:
            salida[particiones[imagen.image_id]].append(imagen.tiempo_s)
    return {k: sorted(v) for k, v in salida.items()}


def _descartados(imagenes) -> list[int]:
    return sorted(int(i.tiempo_s) for i in imagenes if i.descartada_margen)


# --- Orden, mezcla y margen -------------------------------------------------------------


def test_el_reparto_temporal_no_mezcla_tiempos_entre_particiones(fuente_sintetica, tmp_path):
    _, imagenes, particiones = _video(fuente_sintetica, tmp_path)

    t = _tiempos(imagenes, particiones)

    assert max(t["train"]) < min(t["valid"])
    assert max(t["valid"]) < min(t["test"])


def test_el_reparto_respeta_el_orden_aunque_el_zip_este_barajado(fuente_sintetica, tmp_path):
    _, imagenes, particiones = _video(fuente_sintetica, tmp_path)

    t = _tiempos(imagenes, particiones)

    # train son los primeros, valid los siguientes y test los últimos (cortes en 69,5 s y 84,5 s)
    assert t["train"] == list(range(0, 65))
    assert t["valid"] == list(range(75, 80))
    assert t["test"] == list(range(90, 100))


def test_se_descartan_los_fotogramas_a_menos_de_margen_de_cada_corte(fuente_sintetica, tmp_path):
    _, imagenes, particiones = _video(fuente_sintetica, tmp_path)

    assert _descartados(imagenes) == [*range(65, 75), *range(80, 90)]
    assert all(i.image_id not in particiones for i in imagenes if i.descartada_margen)
    t = _tiempos(imagenes, particiones)
    assert min(t["valid"]) - max(t["train"]) >= 2 * 5  # hueco de al menos 2·margen_s
    assert min(t["test"]) - max(t["valid"]) >= 2 * 5


def test_con_margen_cero_no_se_descarta_nada(fuente_sintetica, tmp_path):
    _, imagenes, particiones = _video(fuente_sintetica, tmp_path, temporal=_temporal(margen_s=0))

    assert _descartados(imagenes) == []
    assert len(particiones) == 100
    t = _tiempos(imagenes, particiones)
    assert (len(t["train"]), len(t["valid"]), len(t["test"])) == (70, 15, 15)


def test_el_margen_por_defecto_es_5_segundos(fuente_sintetica, tmp_path):
    temporal = {"patron": PATRON, "fracciones": FRACCIONES}  # sin margen_s
    fuente, imagenes, _ = _video(fuente_sintetica, tmp_path, temporal=temporal)

    assert fuente.particion_temporal.margen_s == 5
    assert _descartados(imagenes) == [*range(65, 75), *range(80, 90)]


def test_el_reparto_es_determinista_y_no_depende_de_la_semilla_ni_del_orden_del_zip(
    fuente_sintetica, tmp_path
):
    def resumen(imagenes):
        return [(i.image_id, i.particion_temporal, i.descartada_margen) for i in imagenes]

    fuente, imagenes, particiones = _video(fuente_sintetica, tmp_path)
    otra_semilla, _ = repartir(imagenes, {"vid": fuente}, 999)

    # Mismo contenido con otro orden dentro del ZIP.
    ruta = fuente_sintetica(
        "vid",
        {"train": list(reversed(_carpetas(100)["train"]))},
        CLASES,
        papel="split",
        particion_temporal=_temporal(),
    )
    imagenes_2 = leer_fuente(cargar_fuente(ruta, tmp_path))

    assert particiones == otra_semilla
    assert resumen(imagenes) == resumen(imagenes_2)


# --- Lectura del tiempo -----------------------------------------------------------------


def test_formula_del_tiempo_en_segundos():
    patron = re.compile(PATRON)

    assert tiempo_de("frame1-02-03-45_jpg.rf.x.jpg", patron, "vid") == pytest.approx(3723.45)
    assert tiempo_de("frame0-00-09-07_jpg.rf.x.jpg", patron, "vid") == pytest.approx(9.07)


def test_un_nombre_sin_patron_da_error_con_el_nombre(fuente_sintetica, tmp_path):
    carpetas = _carpetas(10)
    carpetas["train"].append({"nombre": "raro.jpg", "anotaciones": [("lechuga", RECT)]})
    ruta = fuente_sintetica("vid", carpetas, CLASES, papel="split", particion_temporal=_temporal())

    with pytest.raises(ValueError, match="raro.jpg"):
        leer_fuente(cargar_fuente(ruta, tmp_path))


def test_una_particion_vacia_da_error_claro(fuente_sintetica, tmp_path):
    with pytest.raises(ValueError, match="vacía"):
        _video(fuente_sintetica, tmp_path, temporal=_temporal(margen_s=1000))


# --- Reparto en dos trozos (train y valid, un solo corte) -------------------------------


def test_con_dos_fracciones_hay_un_solo_corte_y_test_queda_vacio(fuente_sintetica, tmp_path):
    temporal = _temporal(fracciones={"train": 0.80, "valid": 0.20})
    fuente, imagenes, particiones = _video(fuente_sintetica, tmp_path, temporal=temporal)

    t = _tiempos(imagenes, particiones)

    # Un solo corte, a 79,5 s: se descartan los fotogramas de 75 a 84 s.
    assert _descartados(imagenes) == list(range(75, 85))
    assert t["train"] == list(range(0, 75))
    assert t["valid"] == list(range(85, 100))
    assert t["test"] == []
    assert min(t["valid"]) - max(t["train"]) >= 2 * 5

    exportar(imagenes, particiones, {"vid": fuente}, tmp_path / "salida")
    assert len(list((tmp_path / "salida" / "train").glob("*.jpg"))) == 75
    assert len(list((tmp_path / "salida" / "valid").glob("*.jpg"))) == 15
    assert list((tmp_path / "salida" / "test").glob("*.jpg")) == []

    informe = construir_informe(imagenes, particiones, 42, [], {"vid": fuente})
    bloque = informe["fuentes"]["vid"]["particion_temporal"]
    assert list(bloque["particiones"]) == ["train", "valid"]
    assert bloque["descartadas_margen"] == 10
    assert informe["totales"]["test"] == {"imagenes": 0, "anotaciones": 0}


def test_con_dos_fracciones_una_particion_vacia_da_error(fuente_sintetica, tmp_path):
    temporal = _temporal(fracciones={"train": 0.80, "valid": 0.20}, margen_s=1000)

    with pytest.raises(ValueError, match="vacía"):
        _video(fuente_sintetica, tmp_path, temporal=temporal)


# --- Validación del YAML ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("extra", "temporal"),
    [
        ({"papel": "split"}, _temporal(patron=r"frame(\d+)_")),  # solo 1 grupo
        ({"papel": "train_only"}, _temporal()),  # solo con papel split
        ({"papel": "split"}, _temporal(fracciones={"train": 0.7, "valid": 0.3, "test": 0.3})),
        ({"papel": "split"}, _temporal(fracciones={"train": 0.7, "test": 0.3})),  # falta valid
        ({"papel": "split"}, _temporal(fracciones={"train": 0.7, "valid": 0.4})),  # suma 1,1
        ({"papel": "split"}, _temporal(fracciones={"train": 1.0, "valid": 0.0})),  # una vacía
        ({"papel": "split"}, _temporal(margen_s=-1)),
        ({"papel": "split", "grupo_regex": r"^(g\d+)_"}, _temporal()),  # dos criterios a la vez
    ],
    ids=["patron_sin_4_grupos", "papel_train_only", "suma_distinta_de_1", "falta_valid",
         "dos_trozos_suman_mas_de_1", "fraccion_cero", "margen_negativo", "con_grupo_regex"],
)  # fmt: skip
def test_yaml_de_particion_temporal_no_valido_da_error(fuente_sintetica, tmp_path, extra, temporal):
    ruta = fuente_sintetica("vid", _carpetas(10), CLASES, particion_temporal=temporal, **extra)

    with pytest.raises(ValueError, match="particion_temporal") as error:
        cargar_fuente(ruta, tmp_path)

    assert "vid" in str(error.value)


# --- Exportación, metadata e informe ----------------------------------------------------


def test_las_descartadas_por_margen_no_se_exportan_ni_tienen_fila(fuente_sintetica, tmp_path):
    fuente, imagenes, particiones = _video(fuente_sintetica, tmp_path)

    exportar(imagenes, particiones, {"vid": fuente}, tmp_path / "salida")

    for particion, esperadas in (("train", 65), ("valid", 5), ("test", 10)):
        assert len(list((tmp_path / "salida" / particion).glob("*.jpg"))) == esperadas
    assert len(filas_metadata(imagenes, TABLA)) == 80  # 100 menos las 20 descartadas


def test_las_excluidas_no_cuentan_en_la_secuencia(fuente_sintetica, tmp_path):
    excluidas = tuple(_nombre(t) for t in range(10))  # los 10 primeros segundos
    _, imagenes, particiones = _video(fuente_sintetica, tmp_path, excluir=excluidas)

    # Quedan 90 imágenes (t = 10…99), repartidas 63 / 14 / 13: los cortes pasan de 69,5 s y
    # 84,5 s a 72,5 s y 86,5 s. Las excluidas no están en ninguna partición ni se descartan.
    assert _descartados(imagenes) == [*range(68, 78), *range(82, 92)]
    en_alguna = [t for ts in _tiempos(imagenes, particiones).values() for t in ts]
    assert min(en_alguna) == 10
    assert not any(i.descartada_margen or i.image_id in particiones for i in imagenes if i.excluida)


def test_el_informe_muestra_rangos_y_descartadas_por_margen(fuente_sintetica, tmp_path):
    fuente, imagenes, particiones = _video(fuente_sintetica, tmp_path)

    informe = construir_informe(imagenes, particiones, 42, [], {"vid": fuente})
    temporal = informe["fuentes"]["vid"]["particion_temporal"]

    assert temporal["margen_s"] == 5
    assert temporal["descartadas_margen"] == 20
    assert temporal["particiones"] == {
        "train": {"imagenes": 65, "desde_s": 0.0, "hasta_s": 64.0},
        "valid": {"imagenes": 5, "desde_s": 75.0, "hasta_s": 79.0},
        "test": {"imagenes": 10, "desde_s": 90.0, "hasta_s": 99.0},
    }
    por_margen = [e for e in informe["excluidas"] if e["motivo"] == "margen temporal"]
    assert len(por_margen) == 20
    texto = formatear_informe(informe)
    assert "valid: 5 imágenes, de 75.00 s a 79.00 s" in texto
    assert "margen temporal" in texto


def test_una_fuente_sin_reparto_temporal_no_tiene_ese_bloque_en_el_informe(
    fuente_sintetica, tmp_path
):
    ruta = fuente_sintetica(
        "otra", {"train": [{"nombre": "a.jpg", "anotaciones": [("lechuga", RECT)]}]}, CLASES
    )
    fuente = cargar_fuente(ruta, tmp_path)
    imagenes = leer_fuente(fuente)
    particiones, avisos = repartir(imagenes, {"otra": fuente}, 42)

    informe = construir_informe(imagenes, particiones, 42, avisos)

    assert "particion_temporal" not in informe["fuentes"]["otra"]


def test_valid_y_test_salen_solo_de_la_fuente_con_reparto_temporal(
    fuente_sintetica, etapa_sintetica, tmp_path
):
    fuente_sintetica(
        "vid",
        _carpetas(100),
        CLASES,
        papel="split",
        particion_temporal=_temporal(),
    )
    fuente_sintetica(
        "otra",
        {"train": [{"nombre": f"{n}.jpg", "anotaciones": [("lechuga", RECT)]} for n in "abcde"]},
        CLASES,
    )  # train_only
    etapa = etapa_sintetica(["vid", "otra"])

    importar(etapa, raiz=tmp_path, metadata=tmp_path / "data" / "metadata.csv", tabla=TABLA)

    salida = tmp_path / "data" / "processed" / "etapa1_v1"
    for particion in ("valid", "test"):
        nombres = [p.name for p in (salida / particion).glob("*.jpg")]
        assert nombres and all(n.startswith("vid_") for n in nombres)
    assert sum(p.name.startswith("otra_") for p in (salida / "train").glob("*.jpg")) == 5


# --- Los YAML reales del repositorio ----------------------------------------------------


def test_los_yaml_reales_del_repositorio_cargan_y_tienen_el_reparto_acordado(tmp_path):
    destino = tmp_path / "configs" / "datasets"
    shutil.copytree(RAIZ_REPO / "configs" / "datasets", destino)
    for ruta in destino.glob("*.yaml"):  # ZIP vacíos de relleno: solo se comprueba que existen
        config = load_config(ruta)
        if "zip" in config:
            assert config["zip"].startswith("data/raw/")
            (tmp_path / config["zip"]).parent.mkdir(parents=True, exist_ok=True)
            (tmp_path / config["zip"]).write_bytes(b"")

    etapa = cargar_etapa(destino / "etapa1_v1.yaml", tmp_path)
    fuentes = {f.nombre: f for f in cargar_fuentes(etapa, tmp_path)}
    tipburn = cargar_fuente(destino / "tipburn.yaml", tmp_path)

    assert etapa.fuentes == ("svodk", "giovi")
    svodk = fuentes["svodk"]
    assert svodk.papel == "split"
    assert svodk.clases == {"krop": "lechuga", "kern": "kern", "kern-krop": "ignorar"}
    assert svodk.particion_temporal.patron.pattern == PATRON
    assert svodk.particion_temporal.fracciones == (0.80, 0.20)  # solo train y valid, un corte
    assert svodk.particion_temporal.margen_s == 5
    assert fuentes["giovi"].papel == "train_only"
    assert tipburn.papel == "train_only"
