"""Pruebas de src/data/dataset_etapa2.py: el dataset de recortes de la etapa 2.

Todo es sintético y vive en ``tmp_path`` (la "raíz"): ZIP COCO con imágenes de manchas de
colores, fotos propias con su metadata.csv y un RF-DETR falso. Nada se descarga ni se lee de data/.
"""

import hashlib
import json
import re
import shutil
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.data.dataset_etapa2 import (
    COLUMNAS,
    cargar_config,
    construir_dataset,
    formatear_informe,
    validar_recortes,
)
from src.data.schema import GROUPS, NO_VISIBLE
from src.evaluation.aggregate import aggregate
from src.experimentos.privacidad import contiene_ruta_absoluta
from src.pipeline.imagenes import leer_imagen

RAIZ_REPO = Path(__file__).resolve().parents[2]
TABLA = load_config(RAIZ_REPO / "configs" / "salida_global.yaml")
SALIDA = Path("data") / "processed" / "etapa2_v1"
TODA = [0, 0, 96, 72]  # caja que ocupa toda la imagen sintética (96x72)

YSSP_CATEGORIAS = ("healthy-Wdl7", "Bacterial", "Downy_mildew", "healthy")
YSSP_CLASES = {
    "healthy": "BIO_OK",
    "Bacterial": "DISEASE_VISIBLE",
    "Downy_mildew": "DISEASE_VISIBLE",
}
PHS_CATEGORIAS = ("disease", "downy_mildew", "growing", "health", "raising_seeding",
                  "sclerotinia_rot")  # fmt: skip
PHS_CLASES = {"health": "BIO_OK", "downy_mildew": "DISEASE_VISIBLE", "sclerotinia_rot": "ROT_DECAY"}


def _img(nombre: str, semilla: int, *cajas, **extra) -> dict:
    return {"nombre": nombre, "semilla": semilla, "cajas": list(cajas), **extra}


def _bloque_yssp(**cambios) -> dict:
    bloque = {"tipo": "coco", "grupo": "daño_bio", "clases": dict(YSSP_CLASES),
              "tope_bio_ok": None, "reparto": None}  # fmt: skip
    bloque.update(cambios)
    return bloque


def _bloque_phs(**cambios) -> dict:
    bloque = {
        "tipo": "coco",
        "grupo": "daño_bio",
        "clases": dict(PHS_CLASES),
        "fases": {"growing": "incluir", "raising_seeding": "excluir"},
        "una_caja_de_estado": True,
        "tope_bio_ok": None,
        "reparto": None,
    }
    bloque.update(cambios)
    return bloque


class _ModeloFalso:
    """Imita ``RFDETR.predict``: devuelve, por orden de llamada, las respuestas dadas."""

    def __init__(self, respuestas):
        self.respuestas = list(respuestas)
        self.llamadas = []

    def predict(self, imagen, threshold=0.5, **kwargs):
        xyxy, confianza = self.respuestas[len(self.llamadas) % len(self.respuestas)]
        self.llamadas.append({"forma": imagen.shape, "threshold": threshold})
        return SimpleNamespace(
            xyxy=np.asarray(xyxy, dtype=float).reshape(-1, 4),
            confidence=np.asarray(confianza, dtype=float),
        )


def _cargador(modelo, registro=None):
    def cargador(ruta, dispositivo):
        if registro is not None:
            registro.append((ruta, dispositivo))
        return modelo

    return cargador


def _construir(ruta, raiz, **opciones) -> dict:
    return construir_dataset(ruta, raiz=raiz, tabla=TABLA, **opciones)


def _csv(raiz: Path) -> pd.DataFrame:
    return pd.read_csv(raiz / SALIDA / "recortes.csv", dtype=str, keep_default_na=False,
                       encoding="utf-8")  # fmt: skip


def _informe(raiz: Path) -> dict:
    return json.loads((raiz / SALIDA / "informe.json").read_text(encoding="utf-8"))


def _huella_carpeta(carpeta: Path) -> dict[str, str]:
    """Contenido de una carpeta: ruta relativa -> sha256 de cada archivo."""
    return {
        p.relative_to(carpeta).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(carpeta.rglob("*"))
        if p.is_file()
    }


# --- El YAML real del repositorio ----------------------------------------------------------


def test_el_yaml_real_carga_con_lo_acordado(tmp_path):
    destino = tmp_path / "configs" / "datasets"
    shutil.copytree(RAIZ_REPO / "configs" / "datasets", destino)
    for ruta in destino.glob("*.yaml"):  # rellenos vacíos: solo se comprueba que existen
        config = load_config(ruta)
        if "zip" in config:
            (tmp_path / config["zip"]).parent.mkdir(parents=True, exist_ok=True)
            (tmp_path / config["zip"]).write_bytes(b"")
    (tmp_path / "data" / "processed" / "propio_piloto").mkdir(parents=True)
    (tmp_path / "data" / "metadata.csv").write_text("", encoding="utf-8")

    config = cargar_config(destino / "etapa2_v1.yaml", tmp_path)

    assert config.salida_rel == "data/processed/etapa2_v1"
    assert (config.recorte.margen, config.recorte.lado_px) == (0.10, 512)
    assert (config.recorte.calidad_jpeg, config.recorte.min_lado_px) == (95, 32)
    assert config.duplicados.umbral_hamming == 4
    assert config.duplicados.giros_y_espejos is True
    assert config.duplicados.prioridad_particiones == ("test", "valid", "train")
    fuentes = {f.nombre: f for f in config.fuentes}
    assert [f.nombre for f in config.fuentes] == ["propio", "yssp", "phs"]
    assert fuentes["propio"].detector.umbral == 0.5
    assert fuentes["propio"].detector.dispositivo == "cpu"
    assert fuentes["yssp"].licencia == "CC BY 4.0"
    assert fuentes["phs"].licencia == "MIT"
    assert fuentes["yssp"].clases == {
        "healthy": "BIO_OK",
        "Bacterial": "DISEASE_VISIBLE",
        "Downy_mildew": "DISEASE_VISIBLE",
        "Lettuce - Anthracnose": "DISEASE_VISIBLE",
        "Powdery_mildew": "DISEASE_VISIBLE",
        "Septoria_Blight": "DISEASE_VISIBLE",
        "lettuce mosaic virus": "DISEASE_VISIBLE",
    }
    assert fuentes["yssp"].reparto is None
    assert fuentes["phs"].clases == PHS_CLASES
    assert fuentes["phs"].fases == {"growing": "incluir", "raising_seeding": "excluir"}
    assert fuentes["phs"].una_caja_de_estado is True
    assert fuentes["phs"].reparto.fracciones == (0.70, 0.15, 0.15)
    regex = fuentes["phs"].reparto.grupo_regex
    nombres = {
        "V006_77_1_09_05_03_12_3_6432e_20201123_25_jpeg.rf.5332c961565ecd876425fae5c6c5b146.jpg":
            "6432e",
        "V006_77_0_00_05_03_12_0_b05_20201204_0166_S01_1_jpg.rf.0281aff1209bf0af128f28c7d07d24a1.jpg":
            "b05",
    }  # fmt: skip
    for nombre, grupo in nombres.items():
        assert regex.search(nombre).group(1) == grupo


# --- Configuración ---------------------------------------------------------------------------


def test_config_con_clave_desconocida_da_error(fuente_etapa2, etapa2_sintetica, tmp_path):
    fuente_etapa2("yssp", {"train": [_img("a.jpg", 1, ("healthy", TODA))]}, YSSP_CATEGORIAS)
    ruta = etapa2_sintetica({"yssp": _bloque_yssp()}, recorte={"margn": 0.1})
    with pytest.raises(ValueError, match="margn"):
        cargar_config(ruta, tmp_path)


def test_config_con_etiqueta_que_no_es_del_grupo(fuente_etapa2, etapa2_sintetica, tmp_path):
    fuente_etapa2("yssp", {"train": [_img("a.jpg", 1, ("healthy", TODA))]}, YSSP_CATEGORIAS)
    clases = {**YSSP_CLASES, "Bacterial": "TORN_LEAF"}  # existe, pero es de daño_fisico
    ruta = etapa2_sintetica({"yssp": _bloque_yssp(clases=clases)})
    with pytest.raises(ValueError, match="TORN_LEAF"):
        cargar_config(ruta, tmp_path)


def test_config_sin_el_yaml_de_la_fuente(etapa2_sintetica, tmp_path):
    ruta = etapa2_sintetica({"yssp": _bloque_yssp()})
    with pytest.raises(FileNotFoundError, match="yssp.yaml"):
        cargar_config(ruta, tmp_path)


def test_config_con_salida_absoluta(fuente_etapa2, etapa2_sintetica, tmp_path):
    fuente_etapa2("yssp", {"train": [_img("a.jpg", 1, ("healthy", TODA))]}, YSSP_CATEGORIAS)
    ruta = etapa2_sintetica({"yssp": _bloque_yssp()}, salida=str(tmp_path / "fuera"))
    with pytest.raises(ValueError, match="salida"):
        cargar_config(ruta, tmp_path)


def test_config_con_prioridad_incompleta(fuente_etapa2, etapa2_sintetica, tmp_path):
    fuente_etapa2("yssp", {"train": [_img("a.jpg", 1, ("healthy", TODA))]}, YSSP_CATEGORIAS)
    ruta = etapa2_sintetica(
        {"yssp": _bloque_yssp()}, duplicados={"prioridad_particiones": ["test", "train"]}
    )
    with pytest.raises(ValueError, match="prioridad_particiones"):
        cargar_config(ruta, tmp_path)


# --- YSSP: mapeo de clases ------------------------------------------------------------------


def test_yssp_mapeo_y_cinco_grupos_no_visibles(fuente_etapa2, etapa2_sintetica, tmp_path):
    fuente_etapa2(
        "yssp",
        {
            "train": [_img("a.jpg", 1, ("healthy", [10, 8, 40, 30])), _img("b.jpg", 2,
                      ("Bacterial", [20.5, 10, 50, 40]))],
            "valid": [_img("c.jpg", 3, ("Downy_mildew", TODA))],
            "test": [_img("d.jpg", 4, ("healthy", TODA))],
        },
        YSSP_CATEGORIAS,
        licencia="CC BY 4.0",
    )  # fmt: skip
    ruta = etapa2_sintetica({"yssp": _bloque_yssp()})

    _construir(ruta, tmp_path)

    tabla = _csv(tmp_path)
    assert list(tabla.columns) == list(COLUMNAS)
    assert tabla["recorte_id"].tolist() == ["yssp_000001", "yssp_000002", "yssp_000003",
                                            "yssp_000004"]  # fmt: skip
    assert tabla["daño_bio"].tolist() == ["BIO_OK", "DISEASE_VISIBLE", "DISEASE_VISIBLE", "BIO_OK"]
    assert tabla["particion"].tolist() == ["train", "train", "valid", "test"]
    assert tabla["etiqueta_original"].tolist() == ["healthy", "Bacterial", "Downy_mildew",
                                                   "healthy"]  # fmt: skip
    for grupo in GROUPS:
        if grupo != "daño_bio":
            assert set(tabla[grupo]) == {NO_VISIBLE}
    for _, fila in tabla.iterrows():
        etiquetas = {g: fila[g] for g in GROUPS}
        assert fila["salida_global"] == aggregate(etiquetas, TABLA).level
    primera = tabla.iloc[0]
    assert primera["imagen_origen"] == "a.jpg"
    assert primera["lechuga_id"] == "yssp_a"  # la imagen de origen: sus recortes no se separan
    assert primera["vista"] == ""
    assert primera["licencia"] == "CC BY 4.0"
    assert primera["origen_caja"] == "anotacion"
    assert [primera[c] for c in ("caja_x", "caja_y", "caja_w", "caja_h")] == ["10", "8", "40", "30"]
    assert tabla.iloc[1]["caja_x"] == "20.5"
    assert re.fullmatch(r"[0-9a-f]{16}", primera["dhash"])


def test_recortes_a_512_sin_deformar_y_sin_exif(fuente_etapa2, etapa2_sintetica, tmp_path):
    fuente_etapa2(
        "yssp",
        {"train": [_img("a.jpg", 1, ("healthy", [10, 10, 40, 20])),
                   _img("b.jpg", 2, ("healthy", [0, 0, 20, 60]))]},
        YSSP_CATEGORIAS,
    )  # fmt: skip
    _construir(etapa2_sintetica({"yssp": _bloque_yssp()}), tmp_path)

    # a: 40x20 + 10 % a cada lado = 48x24 -> 512x256. b: 20x60 en la esquina; con el margen y el
    # corte en el borde queda de x=0 a 22 y de y=0 a 66 -> 22x66 -> 171x512.
    a = leer_imagen(tmp_path / SALIDA / "imagenes" / "yssp_000001.jpg")
    b = leer_imagen(tmp_path / SALIDA / "imagenes" / "yssp_000002.jpg")
    assert a.shape[:2] == (256, 512)
    assert b.shape[:2] == (512, 171)
    for ruta in (tmp_path / SALIDA / "imagenes").glob("*.jpg"):
        assert b"Exif" not in ruta.read_bytes()


def test_clase_con_anotaciones_sin_mapeo_da_error(fuente_etapa2, etapa2_sintetica, tmp_path):
    fuente_etapa2(
        "yssp",
        {"train": [_img("a.jpg", 1, ("healthy", TODA)), _img("b.jpg", 2, ("Powdery", TODA))]},
        (*YSSP_CATEGORIAS, "Powdery"),
    )
    with pytest.raises(ValueError, match="Powdery"):
        _construir(etapa2_sintetica({"yssp": _bloque_yssp()}), tmp_path)
    assert not (tmp_path / SALIDA).exists()


def test_categoria_padre_sin_anotaciones_se_ignora_y_clase_ausente_se_avisa(
    fuente_etapa2, etapa2_sintetica, tmp_path
):
    # "healthy-Wdl7" (padre) no está en el mapeo ni tiene anotaciones: no es un error.
    # "Downy_mildew" está en el mapeo pero no tiene anotaciones: solo un aviso.
    fuente_etapa2("yssp", {"train": [_img("a.jpg", 1, ("healthy", TODA))]}, YSSP_CATEGORIAS)
    informe = _construir(etapa2_sintetica({"yssp": _bloque_yssp()}), tmp_path)
    assert len(_csv(tmp_path)) == 1
    assert any("Downy_mildew" in aviso for aviso in informe["avisos"])
    assert not any("healthy-Wdl7" in aviso for aviso in informe["avisos"])


def test_cajas_pequenas_y_degeneradas_se_descartan_e_informan(
    fuente_etapa2, etapa2_sintetica, tmp_path
):
    fuente_etapa2(
        "yssp",
        {"train": [_img("a.jpg", 1, ("healthy", [0, 0, 40, 40]), ("healthy", [5, 5, 10, 40]),
                        ("Bacterial", [5, 5, 0, 30]), ("healthy", [80, 40, 30, 30]))]},
        YSSP_CATEGORIAS,
    )  # fmt: skip
    ruta = etapa2_sintetica({"yssp": _bloque_yssp()}, recorte={"min_lado_px": 20})

    informe = _construir(ruta, tmp_path)

    tabla = _csv(tmp_path)
    # Se quedan la de 40x40 y la que se sale por la esquina (recortada al borde).
    assert len(tabla) == 2
    motivos = sorted((c["motivo"], c["descartada"]) for c in informe["cajas_raras"])
    assert ("degenerada", True) in motivos
    assert ("pequeña", True) in motivos
    assert any(not descartada for _, descartada in motivos)  # la que se sale: se avisa y se usa
    # Los ids se reparten antes de descartar: las cajas que quedan no se renumeran.
    assert set(tabla["recorte_id"]) <= {f"yssp_{n:06d}" for n in range(1, 5)}


# --- PHS: caja de estado, cajas de fase y plántulas ----------------------------------------


def test_phs_caja_de_estado_fases_y_plantulas(fuente_etapa2, etapa2_sintetica, tmp_path):
    fuente_etapa2(
        "phs",
        {"train": [
            _img("p1.jpg", 1, ("health", [10, 10, 60, 50]), ("growing", [0, 0, 96, 72])),
            _img("p2.jpg", 2, ("health", TODA), ("raising_seeding", TODA)),
            _img("p3.jpg", 3, ("downy_mildew", TODA)),
            _img("p4.jpg", 4, ("sclerotinia_rot", TODA)),
            _img("p5.jpg", 5, ("health", [0, 0, 40, 40]), ("health", [40, 30, 50, 40])),
            _img("p6.jpg", 6, ("growing", TODA)),
        ]},
        PHS_CATEGORIAS,
    )  # fmt: skip
    informe = _construir(etapa2_sintetica({"phs": _bloque_phs()}), tmp_path)

    tabla = _csv(tmp_path)
    assert tabla["imagen_origen"].tolist() == ["p1.jpg", "p3.jpg", "p4.jpg"]
    assert tabla["daño_bio"].tolist() == ["BIO_OK", "DISEASE_VISIBLE", "ROT_DECAY"]
    assert tabla["etiqueta_original"].tolist() == ["health", "downy_mildew", "sclerotinia_rot"]
    p1 = tabla.iloc[0]
    # El recorte es el de la caja de estado, no el de la de fase.
    assert [p1[c] for c in ("caja_x", "caja_y", "caja_w", "caja_h")] == ["10", "10", "60", "50"]
    assert informe["plantulas_excluidas"]["phs"] == {"imagenes": 1, "nombres": ["p2.jpg"]}
    raras = {c["imagen_origen"]: c for c in informe["cajas_raras"]}
    assert "2 cajas de estado" in raras["p5.jpg"]["motivo"]
    assert "0 cajas de estado" in raras["p6.jpg"]["motivo"]
    assert raras["p5.jpg"]["descartada"] and raras["p6.jpg"]["descartada"]


def test_phs_fase_no_declarada_da_error(fuente_etapa2, etapa2_sintetica, tmp_path):
    fuente_etapa2(
        "phs", {"train": [_img("p1.jpg", 1, ("health", TODA), ("growing", TODA))]}, PHS_CATEGORIAS
    )
    ruta = etapa2_sintetica({"phs": _bloque_phs(fases={"raising_seeding": "excluir"})})
    with pytest.raises(ValueError, match="growing"):
        _construir(ruta, tmp_path)


# --- Reparto por grupo (PHS) ----------------------------------------------------------------


GRUPOS = ("g1", "g2", "g3", "g4", "h1", "h2", "h3", "h4")


def _zip_por_grupos(fuente_etapa2):
    """8 grupos (4 con defecto, 4 sanos) de 2 imágenes, desordenados entre carpetas del ZIP."""
    carpetas = {"train": [], "valid": [], "test": []}
    semilla = 0
    for grupo in GRUPOS:
        for n in range(2):
            semilla += 1
            if grupo.startswith("g"):  # g = con defecto
                cajas = [("downy_mildew", TODA)]
            else:  # h = sano, con su caja de fase
                cajas = [("health", TODA), ("growing", TODA)]
            carpeta = ("train", "valid", "test")[semilla % 3]
            carpetas[carpeta].append(_img(f"{grupo}_{n}.jpg", semilla, *cajas))
    fuente_etapa2("phs", carpetas, PHS_CATEGORIAS)


def test_reparto_por_grupo_estratificado_y_estable(fuente_etapa2, etapa2_sintetica, tmp_path):
    _zip_por_grupos(fuente_etapa2)
    reparto = {"grupo_regex": r"^([gh]\d)_", "fracciones": {"train": 0.5, "valid": 0.25,
                                                             "test": 0.25}}  # fmt: skip
    ruta = etapa2_sintetica({"phs": _bloque_phs(reparto=reparto)})

    informe = _construir(ruta, tmp_path)
    primera = _csv(tmp_path)

    assert set(primera["lechuga_id"]) == {f"phs_{g}" for g in GRUPOS}
    # Un grupo nunca cae en dos particiones.
    assert primera.groupby("lechuga_id")["particion"].nunique().max() == 1
    # Cada partición tiene grupos con defecto y grupos sanos.
    for particion in ("train", "valid", "test"):
        de_la_particion = primera[primera["particion"] == particion]
        assert set(de_la_particion["daño_bio"]) == {"BIO_OK", "DISEASE_VISIBLE"}
    bloque = informe["reparto_por_grupo"]["phs"]
    assert sorted(g for p in bloque.values() for g in p["grupos"]) == sorted(
        set(primera["lechuga_id"].str.removeprefix("phs_"))
    )

    _construir(ruta, tmp_path)  # misma semilla: mismo reparto
    assert _csv(tmp_path)["particion"].tolist() == primera["particion"].tolist()


def test_reparto_por_grupo_con_nombre_que_no_encaja(fuente_etapa2, etapa2_sintetica, tmp_path):
    _zip_por_grupos(fuente_etapa2)
    reparto = {"grupo_regex": r"^(z\d)_", "fracciones": {"train": 0.5, "valid": 0.25,
                                                          "test": 0.25}}  # fmt: skip
    with pytest.raises(ValueError, match="grupo_regex"):
        _construir(etapa2_sintetica({"phs": _bloque_phs(reparto=reparto)}), tmp_path)


# --- Tope de BIO_OK ------------------------------------------------------------------------


def _zip_con_sanas(fuente_etapa2):
    train = [_img(f"s{n}.jpg", n, ("healthy", TODA)) for n in range(1, 7)]
    train += [_img("e1.jpg", 20, ("Bacterial", TODA)), _img("e2.jpg", 21, ("Bacterial", TODA))]
    valid = [_img("v1.jpg", 30, ("healthy", TODA)), _img("v2.jpg", 31, ("healthy", TODA))]
    fuente_etapa2("yssp", {"train": train, "valid": valid}, YSSP_CATEGORIAS)


def test_tope_bio_ok_por_fuente_y_particion(fuente_etapa2, etapa2_sintetica, tmp_path):
    _zip_con_sanas(fuente_etapa2)
    sin_tope = etapa2_sintetica({"yssp": _bloque_yssp()})
    _construir(sin_tope, tmp_path)
    todos = _csv(tmp_path)

    con_tope = etapa2_sintetica({"yssp": _bloque_yssp(tope_bio_ok=3)})
    informe = _construir(con_tope, tmp_path)
    tabla = _csv(tmp_path)

    train = tabla[tabla["particion"] == "train"]
    assert (train["daño_bio"] == "BIO_OK").sum() == 3
    assert (train["daño_bio"] == "DISEASE_VISIBLE").sum() == 2  # el tope no toca los defectos
    assert (tabla["particion"] == "valid").sum() == 2  # por debajo del tope: se quedan todas
    assert informe["tope_bio_ok"]["yssp"]["train"] == {"tope": 3, "candidatos": 6, "elegidos": 3}
    # Los ids no se renumeran: cada recorte que queda conserva el que tenía sin tope.
    por_origen = dict(zip(todos["imagen_origen"], todos["recorte_id"], strict=True))
    for origen, recorte_id in zip(tabla["imagen_origen"], tabla["recorte_id"], strict=True):
        assert por_origen[origen] == recorte_id

    _construir(con_tope, tmp_path)  # misma semilla: misma selección
    assert _csv(tmp_path)["recorte_id"].tolist() == tabla["recorte_id"].tolist()


def test_tope_bio_ok_por_particion_y_sin_tope(fuente_etapa2, etapa2_sintetica, tmp_path):
    _zip_con_sanas(fuente_etapa2)
    _construir(etapa2_sintetica({"yssp": _bloque_yssp(tope_bio_ok={"valid": 1})}), tmp_path)
    tabla = _csv(tmp_path)
    assert (tabla["particion"] == "valid").sum() == 1
    assert ((tabla["particion"] == "train") & (tabla["daño_bio"] == "BIO_OK")).sum() == 6


# --- Duplicados ----------------------------------------------------------------------------


def test_duplicado_entre_particiones_se_queda_el_de_evaluacion(
    fuente_etapa2, etapa2_sintetica, tmp_path
):
    fuente_etapa2(
        "yssp",
        {"train": [_img("a.jpg", 1, ("healthy", TODA)), _img("b.jpg", 2, ("healthy", TODA))],
         "valid": [_img("a_copia.jpg", 1, ("healthy", TODA))]},
        YSSP_CATEGORIAS,
    )  # fmt: skip
    informe = _construir(etapa2_sintetica({"yssp": _bloque_yssp()}), tmp_path)

    tabla = _csv(tmp_path)
    assert sorted(tabla["imagen_origen"]) == ["a_copia.jpg", "b.jpg"]
    assert not (tmp_path / SALIDA / "imagenes" / "yssp_000001.jpg").exists()
    duplicados = informe["duplicados"]
    assert duplicados["recortes_quitados"] == 1
    assert duplicados["grupos_que_cruzan"] == 1
    grupo = duplicados["grupos"][0]
    assert grupo["se_queda"] == {"fuente": "yssp", "particion": "valid"}
    quitados = [r["imagen_origen"] for r in grupo["recortes"] if r["se_quita"]]
    assert quitados == ["a.jpg"]


def test_duplicado_entre_fuentes_manda_el_orden_del_yaml(
    fuente_etapa2, etapa2_sintetica, tmp_path
):
    fuente_etapa2("yssp", {"train": [_img("a.jpg", 7, ("healthy", TODA))]}, YSSP_CATEGORIAS)
    fuente_etapa2("otra", {"train": [_img("x.jpg", 7, ("healthy", TODA))]}, YSSP_CATEGORIAS)
    ruta = etapa2_sintetica({"otra": _bloque_yssp(), "yssp": _bloque_yssp()})  # otra va antes

    informe = _construir(ruta, tmp_path)

    assert _csv(tmp_path)["fuente"].tolist() == ["otra"]
    se_queda = informe["duplicados"]["grupos"][0]["se_queda"]
    assert se_queda == {"fuente": "otra", "particion": "train"}


@pytest.mark.parametrize("giros", [True, False])
def test_copia_volteada_se_detecta_solo_con_giros(fuente_etapa2, etapa2_sintetica, tmp_path, giros):
    fuente_etapa2(
        "yssp",
        {"train": [_img("a.jpg", 9, ("healthy", TODA))],
         "test": [_img("a_espejo.jpg", 9, ("healthy", TODA), espejo=True)]},
        YSSP_CATEGORIAS,
    )  # fmt: skip
    ruta = etapa2_sintetica({"yssp": _bloque_yssp()}, duplicados={"giros_y_espejos": giros})
    informe = _construir(ruta, tmp_path)
    esperadas = ["a_espejo.jpg"] if giros else ["a.jpg", "a_espejo.jpg"]
    assert sorted(_csv(tmp_path)["imagen_origen"]) == esperadas
    assert informe["duplicados"]["recortes_quitados"] == (1 if giros else 0)


def test_duplicado_dentro_de_una_particion_se_informa_y_no_se_quita(
    fuente_etapa2, etapa2_sintetica, tmp_path
):
    fuente_etapa2(
        "yssp",
        {"train": [_img("a.jpg", 3, ("healthy", TODA)), _img("a2.jpg", 3, ("healthy", TODA))]},
        YSSP_CATEGORIAS,
    )
    informe = _construir(etapa2_sintetica({"yssp": _bloque_yssp()}), tmp_path)
    assert len(_csv(tmp_path)) == 2
    assert informe["duplicados"]["recortes_quitados"] == 0
    assert informe["duplicados"]["dentro_de_una_celda"]["grupos"] == 1
    assert informe["duplicados"]["dentro_de_una_celda"]["recortes"] == 2


# --- Fotos propias con RF-DETR simulado ----------------------------------------------------

FOTOS = ["LB-001_0_C", "LB-001_0_F", "LB-001_0_G"]


def test_propio_caja_de_mayor_confianza_y_respaldo_de_imagen_entera(
    propio_etapa2, etapa2_sintetica, etiquetas_propias, tmp_path
):
    bloque = propio_etapa2(FOTOS)
    modelo = _ModeloFalso(
        [
            ([[10, 10, 90, 80], [20, 15, 140, 110]], [0.6, 0.9]),  # se elige la de 0,9
            ([[5, 5, 100, 100]], [0.4]),  # por debajo del umbral: como si no hubiera
            (np.zeros((0, 4)), np.zeros(0)),  # sin detección
        ]
    )
    registro = []
    ruta = etapa2_sintetica({"propio": bloque})
    informe = _construir(ruta, tmp_path, cargador_detector=_cargador(modelo, registro))

    tabla = _csv(tmp_path)
    assert tabla["recorte_id"].tolist() == ["propio_000001", "propio_000002", "propio_000003"]
    assert tabla["imagen_origen"].tolist() == [f"{f}.jpg" for f in FOTOS]
    assert tabla["origen_caja"].tolist() == ["rfdetr", "imagen_entera", "imagen_entera"]
    cajas = tabla[["caja_x", "caja_y", "caja_w", "caja_h"]].values.tolist()
    assert cajas == [["20", "15", "120", "95"], ["0", "0", "160", "120"], ["0", "0", "160", "120"]]
    for grupo, etiqueta in etiquetas_propias.items():
        assert set(tabla[grupo]) == {etiqueta}  # copiadas de metadata.csv
    assert set(tabla["lechuga_id"]) == {"LB-001"}
    assert tabla["vista"].tolist() == ["cenital", "lateral_a", "lateral_b"]
    assert set(tabla["particion"]) == {"train"}
    assert set(tabla["licencia"]) == {"propia"}
    assert set(tabla["etiqueta_original"]) == {"tal como viene"}
    assert set(tabla["salida_global"]) == {aggregate(etiquetas_propias, TABLA).level}

    assert registro == [(tmp_path / "runs" / "pesos.pth", "cpu")]  # se carga una sola vez
    assert {llamada["threshold"] for llamada in modelo.llamadas} == {0.5}
    propio = informe["propio"]["propio"]
    assert propio["fotos"] == 3
    assert propio["sin_deteccion"] == ["LB-001_0_F", "LB-001_0_G"]
    assert propio["detecciones"] == [{"image_id": "LB-001_0_C", "confianza": 0.9}]
    assert any("LB-001_0_F" in aviso for aviso in informe["avisos"])

    controles = sorted((tmp_path / SALIDA / "control").glob("*.jpg"))
    assert [c.stem for c in controles] == FOTOS
    assert all(max(leer_imagen(c).shape[:2]) <= 64 for c in controles)


def test_propio_foto_sin_fila_y_fila_sin_foto(propio_etapa2, etapa2_sintetica, tmp_path):
    bloque = propio_etapa2(FOTOS, sin_fila=("LB-001_0_F",), sin_foto=("LB-001_0_G",))
    modelo = _ModeloFalso([(np.zeros((0, 4)), np.zeros(0))])
    with pytest.raises(ValueError) as error:
        _construir(etapa2_sintetica({"propio": bloque}), tmp_path,
                   cargador_detector=_cargador(modelo))  # fmt: skip
    assert "LB-001_0_F" in str(error.value) and "LB-001_0_G" in str(error.value)
    assert not (tmp_path / SALIDA).exists()


def test_propio_sin_pesos_da_error_con_ruta_relativa(propio_etapa2, etapa2_sintetica, tmp_path):
    bloque = propio_etapa2(FOTOS)
    (tmp_path / "runs" / "pesos.pth").unlink()
    with pytest.raises(FileNotFoundError) as error:
        _construir(etapa2_sintetica({"propio": bloque}), tmp_path)
    assert "runs/pesos.pth" in str(error.value)
    assert str(tmp_path) not in str(error.value)


# --- CSV, informe y salida -----------------------------------------------------------------


def _dataset_completo(propio_etapa2, fuente_etapa2, etapa2_sintetica):
    """Propio (2 fotos) + yssp (train, valid y un duplicado) en un solo YAML."""
    bloque = propio_etapa2(FOTOS[:2])
    fuente_etapa2(
        "yssp",
        {"train": [_img("a.jpg", 1, ("healthy", TODA)), _img("b.jpg", 2, ("Bacterial", TODA))],
         "valid": [_img("c.jpg", 1, ("healthy", TODA))]},
        YSSP_CATEGORIAS,
    )  # fmt: skip
    return etapa2_sintetica({"propio": bloque, "yssp": _bloque_yssp()})


def _modelo_una_caja():
    return _ModeloFalso([([[10, 10, 120, 100]], [0.8])])


def test_csv_valido_y_sin_rutas_absolutas(propio_etapa2, fuente_etapa2, etapa2_sintetica, tmp_path):
    ruta = _dataset_completo(propio_etapa2, fuente_etapa2, etapa2_sintetica)
    informe = _construir(ruta, tmp_path, cargador_detector=_cargador(_modelo_una_caja()))

    tabla = _csv(tmp_path)
    assert validar_recortes(tabla) == []
    assert tabla["fuente"].tolist() == ["propio", "propio", "yssp", "yssp"]  # orden del YAML
    texto_csv = (tmp_path / SALIDA / "recortes.csv").read_text(encoding="utf-8")
    texto_informe = (tmp_path / SALIDA / "informe.json").read_text(encoding="utf-8")
    for texto in (texto_csv, texto_informe, formatear_informe(informe)):
        assert not contiene_ruta_absoluta(texto)
        assert str(tmp_path) not in texto and tmp_path.as_posix() not in texto
    assert all("/" not in o and "\\" not in o for o in tabla["imagen_origen"])
    # Un archivo por fila y ninguno de más.
    imagenes = sorted(p.stem for p in (tmp_path / SALIDA / "imagenes").glob("*.jpg"))
    assert imagenes == sorted(tabla["recorte_id"])
    assert informe["totales"] == {"train": 3, "valid": 1, "test": 0}
    etiquetas = informe["recortes"]["yssp"]["train"]["etiquetas"]["daño_bio"]
    assert etiquetas == {"BIO_OK": 0, "ROT_DECAY": 0, "VISIBLE_PEST": 0, "PEST_DAMAGE": 0,
                         "DISEASE_VISIBLE": 1, NO_VISIBLE: 0}  # fmt: skip


def test_validar_recortes_detecta_problemas(propio_etapa2, fuente_etapa2, etapa2_sintetica,
                                            tmp_path):  # fmt: skip
    ruta = _dataset_completo(propio_etapa2, fuente_etapa2, etapa2_sintetica)
    _construir(ruta, tmp_path, cargador_detector=_cargador(_modelo_una_caja()))
    buena = _csv(tmp_path)

    mala = buena.copy()
    mala.loc[0, "color"] = "VERDE"
    mala.loc[1, "particion"] = "val"
    mala.loc[2, "vista"] = "arriba"
    mala.loc[3, "recorte_id"] = mala.loc[2, "recorte_id"]
    mala.loc[0, "imagen_origen"] = "carpeta/x.jpg"
    errores = "\n".join(validar_recortes(mala))
    for trozo in ("VERDE", "'val'", "arriba", "repetido", "carpeta/x.jpg"):
        assert trozo in errores
    assert validar_recortes(buena.drop(columns=["dhash"]))  # falta una columna


def test_el_zip_nunca_se_extrae(fuente_etapa2, etapa2_sintetica, tmp_path, monkeypatch):
    def prohibido(*args, **kwargs):
        raise AssertionError("no se debe extraer nada del ZIP")

    monkeypatch.setattr(zipfile.ZipFile, "extract", prohibido)
    monkeypatch.setattr(zipfile.ZipFile, "extractall", prohibido)
    fuente_etapa2("yssp", {"train": [_img("a.jpg", 1, ("healthy", TODA))]}, YSSP_CATEGORIAS)
    _construir(etapa2_sintetica({"yssp": _bloque_yssp()}), tmp_path)
    assert len(_csv(tmp_path)) == 1


def test_idempotencia(propio_etapa2, fuente_etapa2, etapa2_sintetica, tmp_path):
    ruta = _dataset_completo(propio_etapa2, fuente_etapa2, etapa2_sintetica)
    _construir(ruta, tmp_path, cargador_detector=_cargador(_modelo_una_caja()))
    primera = _huella_carpeta(tmp_path / SALIDA)

    _construir(ruta, tmp_path, cargador_detector=_cargador(_modelo_una_caja()))

    assert _huella_carpeta(tmp_path / SALIDA) == primera
    assert {"recortes.csv", "informe.json"} <= set(primera)
    assert any(k.startswith("control/") for k in primera)


def test_salida_con_archivos_ajenos_no_se_toca(fuente_etapa2, etapa2_sintetica, tmp_path):
    fuente_etapa2("yssp", {"train": [_img("a.jpg", 1, ("healthy", TODA))]}, YSSP_CATEGORIAS)
    ajeno = tmp_path / SALIDA / "mis_notas.txt"
    ajeno.parent.mkdir(parents=True)
    ajeno.write_text("no borrar", encoding="utf-8")
    with pytest.raises(ValueError, match="mis_notas.txt"):
        _construir(etapa2_sintetica({"yssp": _bloque_yssp()}), tmp_path)
    assert ajeno.read_text(encoding="utf-8") == "no borrar"


def test_un_fallo_a_mitad_deja_intacta_la_salida_anterior(
    propio_etapa2, fuente_etapa2, etapa2_sintetica, tmp_path
):
    ruta = _dataset_completo(propio_etapa2, fuente_etapa2, etapa2_sintetica)
    _construir(ruta, tmp_path, cargador_detector=_cargador(_modelo_una_caja()))
    antes = _huella_carpeta(tmp_path / SALIDA)

    class _ModeloQueFalla:
        def predict(self, *args, **kwargs):
            raise RuntimeError("fallo simulado")

    with pytest.raises(RuntimeError, match="fallo simulado"):
        _construir(ruta, tmp_path, cargador_detector=_cargador(_ModeloQueFalla()))

    assert _huella_carpeta(tmp_path / SALIDA) == antes
    assert not (tmp_path / "data" / "processed" / "etapa2_v1.parcial").exists()


def test_comprobar_no_escribe_ni_carga_el_detector(
    propio_etapa2, fuente_etapa2, etapa2_sintetica, tmp_path
):
    ruta = _dataset_completo(propio_etapa2, fuente_etapa2, etapa2_sintetica)

    def no_cargar(ruta, dispositivo):
        raise AssertionError("con comprobar no se carga el detector")

    informe = _construir(ruta, tmp_path, cargador_detector=no_cargar, comprobar=True)

    assert not (tmp_path / SALIDA).exists()
    assert informe["comprobar"] is True
    assert informe["duplicados"] is None  # sin píxeles no se pueden buscar
    assert informe["totales"] == {"train": 4, "valid": 1, "test": 0}  # candidatos, sin duplicados
    assert "comprobar" in formatear_informe(informe)
