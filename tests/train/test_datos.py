"""Pruebas de src/train/datos.py: subconjunto smoke y repeticiones de svodk (copia de trabajo)."""

import hashlib
import json
from pathlib import Path

import pytest

from src.train.datos import crear_copia_de_trabajo, subconjunto_smoke


def _coco(carpeta: Path) -> dict:
    return json.loads((carpeta / "_annotations.coco.json").read_text(encoding="utf-8"))


def _nombres(carpeta: Path) -> list[str]:
    return [i["file_name"] for i in _coco(carpeta)["images"]]


def _huella(carpeta: Path) -> dict[str, str]:
    """Hash de cada archivo de una carpeta: sirve para comprobar que no se modificó nada."""
    return {
        str(p.relative_to(carpeta)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(carpeta.rglob("*"))
        if p.is_file()
    }


def _copiar(dataset, destino, **cambios):
    argumentos = {"repeticiones_svodk": 1, "prefijo_svodk": "svodk_", "smoke_imagenes": None}
    argumentos.update(cambios, semilla=cambios.get("semilla", 42))
    return crear_copia_de_trabajo(dataset, destino, **argumentos)


# --- subconjunto_smoke ------------------------------------------------------------------


def test_el_subconjunto_smoke_tiene_n_imagenes_con_sus_anotaciones(dataset_sintetico):
    coco = _coco(dataset_sintetico() / "train")

    sub = subconjunto_smoke(coco, 4, semilla=42)

    assert len(sub["images"]) == 4
    ids = {i["id"] for i in sub["images"]}
    assert {a["image_id"] for a in sub["annotations"]} == ids  # cada imagen conserva su anotación
    assert len(sub["annotations"]) == 4
    assert sub["categories"] == coco["categories"]
    originales = {i["file_name"] for i in coco["images"]}
    assert {i["file_name"] for i in sub["images"]} <= originales


def test_el_subconjunto_smoke_es_determinista_y_depende_de_la_semilla(dataset_sintetico):
    coco = _coco(dataset_sintetico() / "train")

    a = subconjunto_smoke(coco, 3, semilla=7)
    b = subconjunto_smoke(coco, 3, semilla=7)
    elegidos = {tuple(i["file_name"] for i in subconjunto_smoke(coco, 3, semilla=s)["images"])
                for s in range(8)}  # fmt: skip

    assert a == b
    assert len(elegidos) > 1


def test_si_n_no_es_menor_que_el_total_se_devuelve_todo(dataset_sintetico):
    coco = _coco(dataset_sintetico() / "train")

    assert len(subconjunto_smoke(coco, 8, semilla=1)["images"]) == 8
    assert len(subconjunto_smoke(coco, 500, semilla=1)["images"]) == 8


def test_el_subconjunto_no_modifica_el_json_de_entrada(dataset_sintetico):
    coco = _coco(dataset_sintetico() / "train")
    antes = json.dumps(coco, sort_keys=True)

    subconjunto_smoke(coco, 3, semilla=1)

    assert json.dumps(coco, sort_keys=True) == antes


# --- copia de trabajo: sin repeticiones -------------------------------------------------


def test_con_una_repeticion_la_copia_es_igual_al_original(dataset_sintetico, tmp_path):
    dataset = dataset_sintetico()
    destino = tmp_path / "trabajo" / "e1"

    recuentos = _copiar(dataset, destino)

    assert sorted(_nombres(destino / "train")) == sorted(_nombres(dataset / "train"))
    assert _coco(destino / "train")["annotations"] == _coco(dataset / "train")["annotations"]
    assert recuentos["train"] == {"imagenes": 8, "anotaciones": 8, "imagenes_con_prefijo": 3}
    assert recuentos["valid"] == {"imagenes": 2, "anotaciones": 2, "imagenes_con_prefijo": 2}
    assert recuentos["test"] == {"imagenes": 0, "anotaciones": 0, "imagenes_con_prefijo": 0}
    for nombre in _nombres(destino / "train") + _nombres(destino / "valid"):
        carpeta = "train" if not nombre.endswith(("0100.jpg", "0101.jpg")) else "valid"
        assert (destino / carpeta / nombre).is_file()


def test_la_copia_no_modifica_el_dataset_original(dataset_sintetico, tmp_path):
    dataset = dataset_sintetico()
    antes = _huella(dataset)

    _copiar(dataset, tmp_path / "t" / "e", repeticiones_svodk=3, smoke_imagenes=4)

    assert _huella(dataset) == antes


# --- repeticiones_svodk -----------------------------------------------------------------


def test_con_k_3_svodk_aparece_3_veces_en_train_y_lo_demas_una(dataset_sintetico, tmp_path):
    dataset = dataset_sintetico(n_svodk=3, n_giovi=5)
    destino = tmp_path / "t" / "e"

    recuentos = _copiar(dataset, destino, repeticiones_svodk=3)

    nombres = _nombres(destino / "train")
    assert sum(n.startswith("svodk_") for n in nombres) == 9
    assert sum(n.startswith("giovi_") for n in nombres) == 5
    assert len(nombres) == len(set(nombres)) == 14  # nombres de archivo distintos
    for nombre in nombres:
        assert (destino / "train" / nombre).is_file()
    assert recuentos["train"] == {"imagenes": 14, "anotaciones": 14, "imagenes_con_prefijo": 9}


def test_las_copias_repetidas_tienen_ids_nuevos_y_las_mismas_anotaciones(
    dataset_sintetico, tmp_path
):
    destino = tmp_path / "t" / "e"
    _copiar(dataset_sintetico(n_svodk=2, n_giovi=1), destino, repeticiones_svodk=3)

    coco = _coco(destino / "train")

    ids = [i["id"] for i in coco["images"]]
    assert len(ids) == len(set(ids)) == 7
    ids_anotacion = [a["id"] for a in coco["annotations"]]
    assert len(ids_anotacion) == len(set(ids_anotacion)) == 7
    por_imagen = {i["id"]: i["file_name"] for i in coco["images"]}
    assert {a["image_id"] for a in coco["annotations"]} == set(ids)
    for anotacion in coco["annotations"]:  # todas con la caja y el polígono de la original
        assert anotacion["bbox"] == [2, 2, 8, 6] and anotacion["category_id"] == 1
        assert anotacion["segmentation"] == [[2, 2, 10, 2, 10, 8, 2, 8]]
        assert por_imagen[anotacion["image_id"]]
    # el nombre de las copias deja ver de qué imagen vienen
    assert {n for n in por_imagen.values() if "svodk_0001" in n} == {
        "svodk_0001.jpg", "svodk_0001__r2.jpg", "svodk_0001__r3.jpg"
    }  # fmt: skip


def test_valid_y_test_no_se_repiten_nunca_aunque_sean_de_svodk(dataset_sintetico, tmp_path):
    dataset = dataset_sintetico(n_valid=3)
    destino = tmp_path / "t" / "e"

    recuentos = _copiar(dataset, destino, repeticiones_svodk=5)

    for carpeta in ("valid", "test"):
        assert _coco(destino / carpeta) == _coco(dataset / carpeta)
    assert not any("__r" in n for n in _nombres(destino / "valid"))
    assert not list((destino / "valid").glob("*__r*"))
    assert recuentos["valid"]["imagenes"] == 3  # sigue habiendo 3, no 15


def test_con_repeticiones_y_sin_imagenes_con_el_prefijo_da_error(dataset_sintetico, tmp_path):
    dataset = dataset_sintetico()

    with pytest.raises(ValueError, match="otra_fuente_"):
        _copiar(dataset, tmp_path / "t" / "e", repeticiones_svodk=3, prefijo_svodk="otra_fuente_")


def test_con_una_repeticion_un_prefijo_sin_imagenes_no_es_error(dataset_sintetico, tmp_path):
    _copiar(dataset_sintetico(), tmp_path / "t" / "e", prefijo_svodk="otra_fuente_")  # no lanza


@pytest.mark.parametrize("k", [0, -1, 1.5, "2", True])
def test_repeticiones_no_validas_dan_error(dataset_sintetico, tmp_path, k):
    with pytest.raises(ValueError, match="repeticiones_svodk"):
        _copiar(dataset_sintetico(), tmp_path / "t" / "e", repeticiones_svodk=k)


# --- smoke + repeticiones ---------------------------------------------------------------


def test_el_smoke_deja_n_imagenes_de_train_y_todo_valid(dataset_sintetico, tmp_path):
    dataset = dataset_sintetico()
    destino = tmp_path / "t" / "e"

    recuentos = _copiar(dataset, destino, smoke_imagenes=4)

    assert recuentos["train"]["imagenes"] == 4
    assert len(_nombres(destino / "train")) == 4
    assert len(list((destino / "train").glob("*.jpg"))) == 4  # solo se copian las elegidas
    assert _coco(destino / "valid") == _coco(dataset / "valid")  # todo valid
    assert len(list((destino / "valid").glob("*.jpg"))) == 2


def test_el_smoke_es_determinista(dataset_sintetico, tmp_path):
    dataset = dataset_sintetico()

    _copiar(dataset, tmp_path / "a", smoke_imagenes=4, semilla=5)
    _copiar(dataset, tmp_path / "b", smoke_imagenes=4, semilla=5)

    assert _nombres(tmp_path / "a" / "train") == _nombres(tmp_path / "b" / "train")


def test_el_smoke_con_repeticiones_repite_solo_las_de_svodk_elegidas(dataset_sintetico, tmp_path):
    dataset = dataset_sintetico(n_svodk=3, n_giovi=5)
    sin_repetir = tmp_path / "a"
    repetida = tmp_path / "b"

    _copiar(dataset, sin_repetir, smoke_imagenes=4, semilla=3)
    _copiar(dataset, repetida, smoke_imagenes=4, semilla=3, repeticiones_svodk=3)

    elegidas = _nombres(sin_repetir / "train")
    n_svodk = sum(n.startswith("svodk_") for n in elegidas)
    nombres = _nombres(repetida / "train")
    assert sum(n.startswith("svodk_") for n in nombres) == 3 * n_svodk
    assert len(nombres) == 4 + 2 * n_svodk
    assert len([n for n in nombres if "__r" not in n]) == 4


# --- errores ----------------------------------------------------------------------------


def test_un_destino_que_ya_tiene_contenido_da_error_sin_borrar_nada(dataset_sintetico, tmp_path):
    destino = tmp_path / "t" / "e"
    destino.mkdir(parents=True)
    (destino / "mio.txt").write_text("importante", encoding="utf-8")

    with pytest.raises(ValueError, match="ya existe"):
        _copiar(dataset_sintetico(), destino)

    assert (destino / "mio.txt").read_text(encoding="utf-8") == "importante"


def test_un_dataset_sin_json_de_train_da_error_claro(tmp_path):
    dataset = tmp_path / "vacio"
    dataset.mkdir()

    with pytest.raises(FileNotFoundError, match="train"):
        _copiar(dataset, tmp_path / "t" / "e")
