"""Pruebas de src/data/etapa2.py: la línea de comandos del dataset de recortes de la etapa 2."""

from pathlib import Path

from src.data.etapa2 import main

CATEGORIAS = ("healthy-Wdl7", "Bacterial", "healthy")
BLOQUE = {"tipo": "coco", "grupo": "daño_bio", "clases": {"healthy": "BIO_OK",
          "Bacterial": "DISEASE_VISIBLE"}, "tope_bio_ok": None, "reparto": None}  # fmt: skip
TODA = [0, 0, 96, 72]


def _arbol(carpeta: Path) -> list[str]:
    """Lista ordenada de todo lo que hay bajo una carpeta (nombres y tamaños)."""
    return sorted(f"{p.relative_to(carpeta)}:{p.stat().st_size}" for p in carpeta.rglob("*"))


def _sin_rutas_absolutas(texto: str, raiz: Path) -> bool:
    return str(raiz) not in texto and raiz.as_posix() not in texto


def _preparar(fuente_etapa2, etapa2_sintetica, clase: str = "Bacterial") -> Path:
    fuente_etapa2(
        "yssp",
        {"train": [{"nombre": "a.jpg", "semilla": 1, "cajas": [("healthy", TODA)]},
                   {"nombre": "b.jpg", "semilla": 2, "cajas": [(clase, TODA)]}]},
        (*CATEGORIAS, clase) if clase not in CATEGORIAS else CATEGORIAS,
    )  # fmt: skip
    return etapa2_sintetica({"yssp": dict(BLOQUE)})


def test_comprobar_muestra_el_resumen_y_no_escribe_nada(
    fuente_etapa2, etapa2_sintetica, tmp_path, capsys
):
    ruta = _preparar(fuente_etapa2, etapa2_sintetica)
    antes = _arbol(tmp_path)

    codigo = main(["--config", str(ruta), "--raiz", str(tmp_path), "--comprobar"])

    salida = capsys.readouterr()
    assert codigo == 0
    assert _arbol(tmp_path) == antes
    assert "yssp" in salida.out and "train: 2" in salida.out
    assert "no se ha escrito nada" in salida.out
    assert _sin_rutas_absolutas(salida.out + salida.err, tmp_path)


def test_construye_el_dataset_e_imprime_el_informe(
    fuente_etapa2, etapa2_sintetica, tmp_path, capsys
):
    ruta = _preparar(fuente_etapa2, etapa2_sintetica)

    codigo = main(["--config", str(ruta), "--raiz", str(tmp_path)])

    salida = capsys.readouterr()
    assert codigo == 0
    carpeta = tmp_path / "data" / "processed" / "etapa2_v1"
    assert (carpeta / "recortes.csv").is_file() and (carpeta / "informe.json").is_file()
    assert len(list((carpeta / "imagenes").glob("*.jpg"))) == 2
    assert "duplicados" in salida.out
    assert _sin_rutas_absolutas(salida.out + salida.err, tmp_path)


def test_error_sale_con_codigo_1_y_mensaje_sin_rutas_absolutas(
    fuente_etapa2, etapa2_sintetica, tmp_path, capsys
):
    ruta = _preparar(fuente_etapa2, etapa2_sintetica, clase="Powdery")

    codigo = main(["--config", str(ruta), "--raiz", str(tmp_path)])

    error = capsys.readouterr().err
    assert codigo == 1
    # Antes del error puede haber líneas de avance (también van a stderr): el error es la última.
    assert error.strip().splitlines()[-1].startswith("ERROR:")
    assert "Powdery" in error
    assert _sin_rutas_absolutas(error, tmp_path)
    assert not (tmp_path / "data" / "processed" / "etapa2_v1").exists()


def test_config_inexistente_da_error(tmp_path, capsys):
    codigo = main(["--config", str(tmp_path / "no_existe.yaml"), "--raiz", str(tmp_path)])
    assert codigo == 1
    assert "ERROR:" in capsys.readouterr().err
