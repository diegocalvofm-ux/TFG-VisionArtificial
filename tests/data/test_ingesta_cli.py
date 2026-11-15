"""Pruebas de src/data/ingesta_propia.py: la línea de comandos de la ingesta de fotos propias."""

from pathlib import Path

from src.data.ingesta_propia import main


def _arbol(carpeta: Path) -> list[str]:
    """Lista ordenada de todo lo que hay bajo una carpeta (nombres y tamaños)."""
    return sorted(f"{p.relative_to(carpeta)}:{p.stat().st_size}" for p in carpeta.rglob("*"))


def _fotos(lechuga: str, estado: str) -> list[str]:
    return [f"{lechuga}_{estado}_{v}.jpg" for v in "CFGB"]


def _sin_rutas_absolutas(texto: str, raiz: Path) -> bool:
    return str(raiz) not in texto and raiz.as_posix() not in texto


def test_comprobar_muestra_el_resumen_y_no_escribe_nada(
    ingesta_sintetica, fila_hoja, tmp_path, capsys
):
    ruta = ingesta_sintetica(
        [*_fotos("LB-001", "0"), "LB-001_4_F.jpg"],
        [fila_hoja("LB-001", "0"), fila_hoja("LB-001", "4", amarillas=None)],
    )
    antes = _arbol(tmp_path)

    codigo = main(["--config", str(ruta), "--raiz", str(tmp_path), "--comprobar"])

    salida = capsys.readouterr().out
    assert codigo == 0
    assert _arbol(tmp_path) == antes
    assert "fotos: 5" in salida
    assert "LB-001, estado 4: faltan las vistas C, G, B" in salida
    assert "recuento desconocido" in salida
    assert "no se ha escrito nada" in salida
    assert _sin_rutas_absolutas(salida, tmp_path)


def test_ingesta_escribe_metadata_en_la_ruta_por_defecto(
    ingesta_sintetica, fila_hoja, tmp_path, capsys
):
    ruta = ingesta_sintetica(_fotos("LI-001", "0"), [fila_hoja("LI-001", "0")])

    codigo = main(["--config", str(ruta), "--raiz", str(tmp_path)])

    salida = capsys.readouterr().out
    assert codigo == 0
    assert (tmp_path / "data" / "metadata.csv").is_file()  # <raiz>/data/metadata.csv
    assert (tmp_path / "processed" / "propio" / "informe.json").is_file()
    assert "salida_global" in salida
    assert _sin_rutas_absolutas(salida, tmp_path)


def test_error_sale_con_codigo_1_y_mensaje_sin_rutas_absolutas(
    ingesta_sintetica, fila_hoja, tmp_path, capsys
):
    ruta = ingesta_sintetica([*_fotos("LI-001", "0"), "LI-002_0_C.jpg"], [fila_hoja("LI-001", "0")])

    codigo = main(["--config", str(ruta), "--raiz", str(tmp_path)])

    error = capsys.readouterr().err
    assert codigo == 1
    assert error.startswith("ERROR:")
    assert "LI-002_0_C.jpg" in error
    assert _sin_rutas_absolutas(error, tmp_path)
    assert not (tmp_path / "data" / "metadata.csv").exists()


def test_config_inexistente_da_error(tmp_path, capsys):
    codigo = main(["--config", str(tmp_path / "no_existe.yaml"), "--raiz", str(tmp_path)])
    assert codigo == 1
    assert "ERROR:" in capsys.readouterr().err
