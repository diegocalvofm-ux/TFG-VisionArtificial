"""Pruebas de src/data/publicos.py: importación de datasets públicos COCO (ZIP sintéticos)."""

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from src.config import load_config
from src.data.publicos import (
    PARTICION_SALIDA,
    cargar_etapa,
    cargar_fuente,
    cargar_fuentes,
    construir_informe,
    escribir_metadata,
    exportar,
    filas_metadata,
    fusionar_metadata,
    importar,
    leer_fuente,
    repartir,
)
from src.data.schema import GROUPS, METADATA_COLUMNS, NO_VISIBLE, validate_metadata
from src.evaluation.aggregate import aggregate

RAIZ_REPO = Path(__file__).resolve().parents[2]
TABLA = load_config(RAIZ_REPO / "configs" / "salida_global.yaml")

# Rectángulo de (2, 2) a (10, 8): caja [2, 2, 8, 6].
RECT = [[2, 2, 10, 2, 10, 8, 2, 8]]
FRACCIONES = {"train": 0.5, "valid": 0.25, "test": 0.25}
CLASES_BASICAS = {"lechuga": "lechuga", "kern": "kern"}


def _carpeta(nombres, clase="lechuga", poligonos=RECT):
    """Imágenes con una anotación de ``clase`` cada una."""
    return [{"nombre": n, "anotaciones": [(clase, poligonos)]} for n in nombres]


def _grupos(n_grupos=12, por_grupo=2):
    """Nombres ``g00_0.jpg``, ``g00_1.jpg``... (el grupo se saca con ``^(g\\d+)_``)."""
    return [f"g{g:02d}_{k}.jpg" for g in range(n_grupos) for k in range(por_grupo)]


def _fuente_split(fuente_sintetica, nombre="pub", **extra):
    return fuente_sintetica(
        nombre,
        {"train": _carpeta(_grupos())},
        CLASES_BASICAS,
        papel="split",
        fracciones=FRACCIONES,
        grupo_regex=r"^(g\d+)_",
        **extra,
    )


def _coco(salida: Path, particion: str) -> dict:
    return json.loads((salida / particion / "_annotations.coco.json").read_text(encoding="utf-8"))


def _preparar(ruta_yaml, raiz, semilla=42):
    """Carga la fuente, lee sus imágenes y las reparte. Devuelve (fuente, imágenes, particiones)."""
    fuente = cargar_fuente(ruta_yaml, raiz)
    imagenes = leer_fuente(fuente)
    particiones, _ = repartir(imagenes, {fuente.nombre: fuente}, semilla)
    return fuente, imagenes, particiones


# --- Mapeo de clases --------------------------------------------------------------------


def test_clase_sin_mapeo_da_error_que_la_nombra(fuente_sintetica, tmp_path):
    ruta = fuente_sintetica(
        "pub",
        {"train": [{"nombre": "a.jpg", "anotaciones": [("kern", RECT)]}]},
        {"lechuga": "lechuga"},  # falta "kern"
    )

    with pytest.raises(ValueError, match="kern") as error:
        leer_fuente(cargar_fuente(ruta, tmp_path))

    assert "pub" in str(error.value)


def test_categoria_padre_sin_anotaciones_no_exige_mapeo(fuente_sintetica, tmp_path):
    # "padre" y "kern" no tienen anotaciones ni mapeo.
    ruta = fuente_sintetica("pub", {"train": _carpeta(["a.jpg"])}, {"lechuga": "lechuga"})

    imagenes = leer_fuente(cargar_fuente(ruta, tmp_path))

    assert len(imagenes) == 1


def test_destino_no_valido_en_el_yaml_da_error(fuente_sintetica, tmp_path):
    ruta = fuente_sintetica("pub", {"train": _carpeta(["a.jpg"])}, {"lechuga": "planta"})

    with pytest.raises(ValueError, match="planta"):
        cargar_fuente(ruta, tmp_path)


def test_ignorar_kern_y_tipburn_no_se_exportan_pero_quedan_en_etiqueta_original(
    fuente_sintetica, tmp_path
):
    ruta = fuente_sintetica(
        "pub",
        {
            "train": [
                {
                    "nombre": "a.jpg",
                    "anotaciones": [
                        ("lechuga", RECT),
                        ("kern", RECT),
                        ("tipburn", RECT),
                        ("fondo", RECT),
                    ],
                }
            ]
        },
        {"lechuga": "lechuga", "kern": "kern", "tipburn": "tipburn", "fondo": "ignorar"},
        categorias=("padre", "lechuga", "kern", "tipburn", "fondo"),
    )
    fuente, imagenes, particiones = _preparar(ruta, tmp_path)

    exportar(imagenes, particiones, {"pub": fuente}, tmp_path / "salida")

    anotaciones = _coco(tmp_path / "salida", "train")["annotations"]
    assert len(anotaciones) == 1  # solo la lechuga
    assert filas_metadata(imagenes, TABLA)["etiqueta_original"].tolist() == [
        "fondo;kern;lechuga;tipburn"
    ]


# --- Exclusiones e imágenes sin lechuga -------------------------------------------------


def test_excluir_imagenes_por_nombre_y_por_ruta(fuente_sintetica, tmp_path):
    ruta = fuente_sintetica(
        "pub",
        {"train": _carpeta(["a.jpg", "b.jpg", "c.jpg"])},
        CLASES_BASICAS,
        excluir=("a.jpg", "train/b.jpg"),
    )
    fuente, imagenes, particiones = _preparar(ruta, tmp_path)

    exportar(imagenes, particiones, {"pub": fuente}, tmp_path / "salida")

    nombres = [i["file_name"] for i in _coco(tmp_path / "salida", "train")["images"]]
    assert nombres == ["pub_0003.jpg"]
    assert filas_metadata(imagenes, TABLA)["archivo_original"].tolist() == ["train/c.jpg"]


def test_excluir_una_imagen_que_no_existe_da_error(fuente_sintetica, tmp_path):
    ruta = fuente_sintetica(
        "pub", {"train": _carpeta(["a.jpg"])}, CLASES_BASICAS, excluir=("no_esta.jpg",)
    )

    with pytest.raises(ValueError, match="no_esta.jpg"):
        leer_fuente(cargar_fuente(ruta, tmp_path))


def test_imagen_sin_lechuga_no_se_exporta_pero_tiene_fila_y_sale_en_el_informe(
    fuente_sintetica, tmp_path
):
    carpetas = {
        "train": [
            *_carpeta(["a.jpg"]),
            {"nombre": "solo_kern.jpg", "anotaciones": [("kern", RECT)]},
            {"nombre": "vacia.jpg", "anotaciones": []},
        ]
    }
    ruta = fuente_sintetica("pub", carpetas, CLASES_BASICAS)
    fuente, imagenes, particiones = _preparar(ruta, tmp_path)

    exportar(imagenes, particiones, {"pub": fuente}, tmp_path / "salida")
    informe = construir_informe(imagenes, particiones, 42, [])

    assert len(_coco(tmp_path / "salida", "train")["images"]) == 1
    assert len(filas_metadata(imagenes, TABLA)) == 3
    assert sorted(i["archivo_original"] for i in informe["sin_lechuga"]) == [
        "train/solo_kern.jpg",
        "train/vacia.jpg",
    ]


# --- Particiones ------------------------------------------------------------------------


def test_train_only_nunca_va_a_valid_ni_a_test(fuente_sintetica, tmp_path):
    ruta = fuente_sintetica(  # aunque Roboflow la trajera en valid y test
        "pub",
        {"train": _carpeta(["a.jpg"]), "valid": _carpeta(["b.jpg"]), "test": _carpeta(["c.jpg"])},
        CLASES_BASICAS,
        papel="train_only",
    )
    fuente, imagenes, particiones = _preparar(ruta, tmp_path)

    exportar(imagenes, particiones, {"pub": fuente}, tmp_path / "salida")

    assert set(particiones.values()) == {"train"}
    assert len(_coco(tmp_path / "salida", "train")["images"]) == 3
    assert _coco(tmp_path / "salida", "valid")["images"] == []
    assert _coco(tmp_path / "salida", "test")["images"] == []


def test_ningun_grupo_en_dos_particiones(fuente_sintetica, tmp_path):
    fuente, imagenes, particiones = _preparar(_fuente_split(fuente_sintetica), tmp_path)

    por_grupo: dict[str, set] = {}
    for imagen in imagenes:
        por_grupo.setdefault(imagen.lechuga_id, set()).add(particiones[imagen.image_id])
    assert all(len(p) == 1 for p in por_grupo.values())
    assert set(particiones.values()) == set(PARTICION_SALIDA.values())  # train, valid y test


def test_misma_semilla_mismas_particiones_y_mismo_json(fuente_sintetica, tmp_path):
    fuente, imagenes, _ = _preparar(_fuente_split(fuente_sintetica), tmp_path)
    fuentes = {"pub": fuente}

    a, _ = repartir(imagenes, fuentes, 7)
    b, _ = repartir(imagenes, fuentes, 7)
    otra, _ = repartir(imagenes, fuentes, 8)
    exportar(imagenes, a, fuentes, tmp_path / "s1")
    exportar(imagenes, b, fuentes, tmp_path / "s2")

    assert a == b
    assert a != otra
    for particion in ("train", "valid", "test"):
        texto1 = (tmp_path / "s1" / particion / "_annotations.coco.json").read_text("utf-8")
        texto2 = (tmp_path / "s2" / particion / "_annotations.coco.json").read_text("utf-8")
        assert texto1 == texto2


def test_grupo_regex_pone_prefijo_de_la_fuente(fuente_sintetica, tmp_path):
    fuente = cargar_fuente(_fuente_split(fuente_sintetica), tmp_path)

    imagenes = leer_fuente(fuente)

    assert {i.lechuga_id for i in imagenes} == {f"pub_g{g:02d}" for g in range(12)}


def test_nombre_que_no_encaja_con_grupo_regex_da_error_con_el_archivo(fuente_sintetica, tmp_path):
    ruta = fuente_sintetica(
        "pub",
        {"train": _carpeta(["raro.jpg"])},
        CLASES_BASICAS,
        papel="split",
        fracciones=FRACCIONES,
        grupo_regex=r"^(g\d+)_",
    )

    with pytest.raises(ValueError, match="raro.jpg"):
        leer_fuente(cargar_fuente(ruta, tmp_path))


def test_sin_grupo_regex_lechuga_id_va_vacio_y_el_aviso_llega_al_informe(
    fuente_sintetica, tmp_path
):
    ruta = fuente_sintetica(
        "pub",
        {"train": _carpeta([f"i{n:02d}.jpg" for n in range(12)])},
        CLASES_BASICAS,
        papel="split",
        fracciones=FRACCIONES,
    )
    fuente = cargar_fuente(ruta, tmp_path)
    imagenes = leer_fuente(fuente)

    particiones, avisos = repartir(imagenes, {"pub": fuente}, 42)
    informe = construir_informe(imagenes, particiones, 42, avisos)

    assert filas_metadata(imagenes, TABLA)["lechuga_id"].tolist() == [""] * 12
    assert len(informe["avisos"]) == 1
    assert "sin lechuga_id" in informe["avisos"][0]


# --- Exportación ------------------------------------------------------------------------


def test_cajas_desde_poligonos_y_poligonos_conservados(fuente_sintetica, tmp_path):
    triangulo = [[0, 0, 4, 0, 2, 6]]
    carpetas = {
        "train": [{"nombre": "a.jpg", "anotaciones": [("lechuga", RECT), ("lechuga", triangulo)]}]
    }
    ruta = fuente_sintetica("pub", carpetas, CLASES_BASICAS)
    fuente, imagenes, particiones = _preparar(ruta, tmp_path)

    exportar(imagenes, particiones, {"pub": fuente}, tmp_path / "salida")

    coco = _coco(tmp_path / "salida", "train")
    assert coco["categories"] == [{"id": 1, "name": "lechuga"}]
    assert [a["bbox"] for a in coco["annotations"]] == [[2, 2, 8, 6], [0, 0, 4, 6]]  # no [0,0,1,1]
    assert [a["area"] for a in coco["annotations"]] == [48, 12]
    assert [a["segmentation"] for a in coco["annotations"]] == [RECT, triangulo]
    assert {a["category_id"] for a in coco["annotations"]} == {1}
    assert coco["images"][0]["file_name"] == "pub_0001.jpg"
    assert (tmp_path / "salida" / "train" / "pub_0001.jpg").is_file()


def test_poligono_degenerado_se_descarta_y_se_cuenta(fuente_sintetica, tmp_path):
    carpetas = {
        "train": [
            {"nombre": "a.jpg", "anotaciones": [("lechuga", RECT), ("lechuga", [[1, 1, 2, 2]])]},
        ]
    }
    ruta = fuente_sintetica("pub", carpetas, CLASES_BASICAS)

    imagenes = leer_fuente(cargar_fuente(ruta, tmp_path))

    assert len(imagenes[0].anotaciones) == 1
    assert imagenes[0].descartadas == 1


def test_salida_anterior_se_borra(fuente_sintetica, tmp_path):
    ruta = fuente_sintetica("pub", {"train": _carpeta(["a.jpg"])}, CLASES_BASICAS)
    fuente, imagenes, particiones = _preparar(ruta, tmp_path)
    salida = tmp_path / "salida"
    (salida / "train").mkdir(parents=True)
    (salida / "train" / "viejo.jpg").write_bytes(b"x")
    (salida / "informe.json").write_text("{}", encoding="utf-8")

    exportar(imagenes, particiones, {"pub": fuente}, salida)

    assert not (salida / "train" / "viejo.jpg").exists()
    assert (salida / "train" / "pub_0001.jpg").is_file()


def test_salida_con_contenido_ajeno_se_rechaza_sin_borrar(fuente_sintetica, tmp_path):
    ruta = fuente_sintetica("pub", {"train": _carpeta(["a.jpg"])}, CLASES_BASICAS)
    fuente, imagenes, particiones = _preparar(ruta, tmp_path)
    salida = tmp_path / "salida"
    salida.mkdir()
    (salida / "mis_notas.txt").write_text("importante", encoding="utf-8")

    with pytest.raises(ValueError, match="mis_notas.txt"):
        exportar(imagenes, particiones, {"pub": fuente}, salida)

    assert (salida / "mis_notas.txt").read_text(encoding="utf-8") == "importante"


def test_el_zip_original_no_se_modifica(fuente_sintetica, tmp_path):
    ruta = _fuente_split(fuente_sintetica)
    zip_original = tmp_path / "zips" / "pub.zip"
    antes = hashlib.sha256(zip_original.read_bytes()).hexdigest()
    carpeta_antes = sorted(p.name for p in zip_original.parent.iterdir())
    fuente, imagenes, particiones = _preparar(ruta, tmp_path)

    exportar(imagenes, particiones, {"pub": fuente}, tmp_path / "salida")

    assert hashlib.sha256(zip_original.read_bytes()).hexdigest() == antes
    assert sorted(p.name for p in zip_original.parent.iterdir()) == carpeta_antes


# --- Metadatos --------------------------------------------------------------------------


def test_filas_generadas_son_validas_y_tienen_las_columnas_del_esquema(fuente_sintetica, tmp_path):
    fuente = cargar_fuente(_fuente_split(fuente_sintetica), tmp_path)
    imagenes = leer_fuente(fuente)

    df = filas_metadata(imagenes, TABLA)

    assert tuple(df.columns) == METADATA_COLUMNS
    assert validate_metadata(df) == []
    assert (df[list(GROUPS)] == NO_VISIBLE).all().all()
    nivel = aggregate(dict.fromkeys(GROUPS, NO_VISIBLE), TABLA).level
    assert (df["salida_global"] == nivel).all()
    assert (df["fuente"] == "pub").all() and (df["licencia"] == "MIT").all()


def test_image_id_corto_unico_y_estable_al_excluir_otra_imagen(fuente_sintetica, tmp_path):
    carpetas = {"train": _carpeta(["a.jpg", "b.jpg", "c.jpg"])}
    ruta = fuente_sintetica("pub", carpetas, CLASES_BASICAS)
    ids_completos = {
        i.archivo_original: i.image_id for i in leer_fuente(cargar_fuente(ruta, tmp_path))
    }
    ruta = fuente_sintetica("pub", carpetas, CLASES_BASICAS, excluir=("a.jpg",))

    df = filas_metadata(leer_fuente(cargar_fuente(ruta, tmp_path)), TABLA)

    assert ids_completos == {
        "train/a.jpg": "pub_0001",
        "train/b.jpg": "pub_0002",
        "train/c.jpg": "pub_0003",
    }
    assert dict(zip(df["archivo_original"], df["image_id"], strict=True)) == {
        "train/b.jpg": "pub_0002",
        "train/c.jpg": "pub_0003",
    }


def test_fusionar_reemplaza_solo_las_filas_de_las_fuentes_importadas(
    fuente_sintetica, metadata_valida, tmp_path
):
    ruta = fuente_sintetica("pub", {"train": _carpeta(["a.jpg", "b.jpg"])}, CLASES_BASICAS)
    nuevas = filas_metadata(leer_fuente(cargar_fuente(ruta, tmp_path)), TABLA)
    propias = metadata_valida(n_lechugas=2, fotos_por_lechuga=1)  # fuente "propia"

    una_vez = fusionar_metadata(propias, nuevas, ["pub"])
    dos_veces = fusionar_metadata(una_vez, nuevas, ["pub"])

    assert len(una_vez) == 4 and len(dos_veces) == 4  # no duplica
    assert dos_veces["fuente"].tolist().count("propia") == 2
    assert validate_metadata(dos_veces) == []


def test_escribir_metadata_no_escribe_si_no_valida(tmp_path, metadata_valida):
    df = metadata_valida(n_lechugas=1, fotos_por_lechuga=1)
    df.loc[0, "licencia"] = ""
    destino = tmp_path / "metadata.csv"

    with pytest.raises(ValueError, match="licencia"):
        escribir_metadata(df, destino)

    assert not destino.exists()


# --- Orquestación e informe -------------------------------------------------------------


def test_importar_de_extremo_a_extremo_no_duplica_filas_y_escribe_el_informe(
    fuente_sintetica, etapa_sintetica, tmp_path
):
    _fuente_split(fuente_sintetica, "pub")
    fuente_sintetica(
        "fotogramas",
        {
            "train": _carpeta(["f1.jpg", "f2.jpg"], clase="kern")
            + _carpeta(["f3.jpg"], clase="lechuga")
        },
        CLASES_BASICAS,
    )
    etapa = etapa_sintetica(["pub", "fotogramas"])
    metadata = tmp_path / "data" / "metadata.csv"
    metadata.parent.mkdir(parents=True)
    kwargs = {"raiz": tmp_path, "metadata": metadata, "tabla": TABLA}

    informe = importar(etapa, **kwargs)
    primera = pd.read_csv(metadata, dtype=str, keep_default_na=False, encoding="utf-8")
    importar(etapa, **kwargs)
    segunda = pd.read_csv(metadata, dtype=str, keep_default_na=False, encoding="utf-8")

    assert len(primera) == len(segunda) == 24 + 3
    assert validate_metadata(segunda) == []
    en_disco = json.loads((tmp_path / "data/processed/etapa1_v1/informe.json").read_text("utf-8"))
    assert en_disco == informe
    por_particion = informe["fuentes"]["pub"]["particiones"]
    assert sum(p["imagenes"] for p in por_particion.values()) == 24
    assert sum(p["anotaciones"] for p in por_particion.values()) == 24
    assert informe["fuentes"]["fotogramas"]["particiones"]["train"] == {
        "imagenes": 1,
        "anotaciones": 1,
    }
    assert len(informe["sin_lechuga"]) == 2  # f1 y f2 solo tienen kern
    assert str(tmp_path) not in json.dumps(informe)  # sin rutas absolutas


def test_importar_no_escribe_metadata_si_la_exportacion_se_rechaza(
    fuente_sintetica, etapa_sintetica, tmp_path
):
    fuente_sintetica("pub", {"train": _carpeta(["a.jpg"])}, CLASES_BASICAS)
    etapa = etapa_sintetica(["pub"])
    salida = tmp_path / "data" / "processed" / "etapa1_v1"
    salida.mkdir(parents=True)
    (salida / "ajeno.txt").write_text("x", encoding="utf-8")
    metadata = tmp_path / "data" / "metadata.csv"

    with pytest.raises(ValueError, match="ajeno.txt"):
        importar(etapa, raiz=tmp_path, metadata=metadata, tabla=TABLA)

    assert not metadata.exists()


def test_cargar_etapa_y_fuentes_resuelven_rutas_desde_la_raiz(
    fuente_sintetica, etapa_sintetica, tmp_path
):
    fuente_sintetica("pub", {"train": _carpeta(["a.jpg"])}, CLASES_BASICAS)
    etapa = cargar_etapa(etapa_sintetica(["pub"], semilla=5), tmp_path)

    fuentes = cargar_fuentes(etapa, tmp_path)

    assert etapa.semilla == 5
    assert etapa.salida == tmp_path / "data" / "processed" / "etapa1_v1"
    assert [f.nombre for f in fuentes] == ["pub"]
    assert fuentes[0].zip == tmp_path / "zips" / "pub.zip"
