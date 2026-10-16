"""Pruebas de src/experimentos/ejecuciones.py: resumir los JSON de una ejecución."""

import pytest

from src.data.schema import GLOBAL_LEVELS
from src.experimentos.ejecuciones import (
    COLUMNAS,
    ResumenEjecucion,
    actualizar_filas,
    contenido_bloque,
    fila_ejecucion,
    parsear_tabla,
    renderizar_tabla,
    resumir_carpeta,
)

# Nombre con el que se avisa de cada dato que falta en un JSON.
CAMPOS_DEL_JSON = (
    "env.detector", "env.device", "env.commit", "env.dirty", "image_size", "rss_mb",
    "model_load_ms", "timing_ms.repeticiones", "timing_ms.calentamiento",
)  # fmt: skip

CAMPOS_OPCIONALES = (
    "detector", "dispositivo", "commit", "sucio", "tamano_imagen", "repeticiones",
    "calentamiento", "memoria_max_mb", "carga_modelo_ms",
)  # fmt: skip


def _resumen(**cambios):
    datos = {
        "n_imagenes": 3,
        "ignorados": (),
        "mediana_total_ms": 20.0,
        "mediana_detect_ms": 10.0,
        "lechugas": 3,
        "sin_detecciones": 1,
        "niveles": {"ACCEPT": 1, "REVIEW": 2, "NON-CONFORMING": 0},
    }
    return ResumenEjecucion(**{**datos, **cambios})


def test_las_columnas_de_la_tabla():
    assert COLUMNAS == (
        "Carpeta", "Detector", "Dispositivo", "Commit", "Sucio", "Imágenes", "Tamaño imagen",
        "Repeticiones", "Calentamiento", "Mediana (ms)", "Mediana detección (ms)",
        "Carga modelo (ms)", "Memoria (MB)", "Lechugas", "Sin detección", *GLOBAL_LEVELS,
        "Observación",
    )  # fmt: skip


# --- resumir_carpeta --------------------------------------------------------------------


def test_mediana_con_numero_impar_de_imagenes(tmp_path, escribir_json):
    for i, total in enumerate((30, 10, 20)):
        escribir_json(tmp_path, f"f{i}", total_ms=total, detect_ms=total / 2)

    resumen = resumir_carpeta(tmp_path)

    assert resumen.n_imagenes == 3
    assert resumen.mediana_total_ms == 20 and resumen.mediana_detect_ms == 10


def test_mediana_con_numero_par_de_imagenes_promedia_las_dos_centrales(tmp_path, escribir_json):
    for i, total in enumerate((10, 20, 30, 41)):
        escribir_json(tmp_path, f"f{i}", total_ms=total)

    assert resumir_carpeta(tmp_path).mediana_total_ms == 25


def test_cuenta_lechugas_imagenes_sin_detecciones_y_niveles(tmp_path, escribir_json):
    escribir_json(tmp_path, "a", niveles=("REVIEW", "ACCEPT"))
    escribir_json(tmp_path, "b", niveles=("REVIEW",))
    escribir_json(tmp_path, "c", niveles=())

    resumen = resumir_carpeta(tmp_path)

    assert resumen.lechugas == 3 and resumen.sin_detecciones == 1
    assert resumen.niveles == {"ACCEPT": 1, "REVIEW": 2, "NON-CONFORMING": 0}


def test_los_json_invalidos_se_cuentan_aparte_y_no_se_ignoran_en_silencio(tmp_path, escribir_json):
    escribir_json(tmp_path, "bien")
    (tmp_path / "rota.json").write_bytes(b"esto no es json")
    (tmp_path / "lista.json").write_text("[1, 2]", encoding="utf-8")
    (tmp_path / "sin_tiempos.json").write_text('{"lettuces": []}', encoding="utf-8")
    (tmp_path / "tiempo_texto.json").write_text(
        '{"lettuces": [], "timing_ms": {"detect": "x", "total": 1}}', encoding="utf-8"
    )
    (tmp_path / "tiempo_bool.json").write_text(
        '{"lettuces": [], "timing_ms": {"detect": true, "total": 1}}', encoding="utf-8"
    )
    (tmp_path / "nivel_raro.json").write_text(
        '{"lettuces": [{"global": "RARO"}], "timing_ms": {"detect": 1, "total": 1}}',
        encoding="utf-8",
    )

    resumen = resumir_carpeta(tmp_path)

    assert resumen.n_imagenes == 1
    assert resumen.ignorados == (
        "lista.json", "nivel_raro.json", "rota.json", "sin_tiempos.json", "tiempo_bool.json",
        "tiempo_texto.json",
    )  # fmt: skip


def test_solo_lee_json_de_esa_carpeta(tmp_path, escribir_json):
    escribir_json(tmp_path, "a")
    escribir_json(tmp_path / "sub", "b")  # una subcarpeta no cuenta
    (tmp_path / "foto.jpg").write_bytes(b"x")
    (tmp_path / "notas.txt").write_text("hola", encoding="utf-8")

    resumen = resumir_carpeta(tmp_path)

    assert resumen.n_imagenes == 1 and resumen.ignorados == ()


def test_carpeta_sin_json_validos_lanza_value_error_claro(tmp_path):
    (tmp_path / "rota.json").write_bytes(b"{")

    with pytest.raises(ValueError, match="JSON válidos"):
        resumir_carpeta(tmp_path)


def test_carpeta_inexistente_lanza_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError, match="no_existe"):
        resumir_carpeta(tmp_path / "no_existe")


# --- JSON con y sin los campos de la tarea 5 --------------------------------------------


def test_un_json_antiguo_deja_todo_sin_dato_y_cuenta_lo_que_falta(tmp_path, escribir_json):
    for nombre in ("a", "b", "c"):
        escribir_json(tmp_path, nombre)  # como los de antes de la tarea 5

    resumen = resumir_carpeta(tmp_path)

    assert all(getattr(resumen, campo) is None for campo in CAMPOS_OPCIONALES)
    assert resumen.faltantes == dict.fromkeys(CAMPOS_DEL_JSON, 3)  # falta en los 3 de los 3
    assert resumen.avisos == ()  # los avisos de formato son aparte; esto lo cuenta "faltantes"
    assert resumen.commits_distintos is False and resumen.alguno_sucio is False


def test_un_json_completo_se_lee_entero_y_no_falta_nada(tmp_path, escribir_json):
    for i, (rss, tamano) in enumerate(((100.0, [640, 480]), (300.5, [640, 480]))):
        escribir_json(
            tmp_path, f"f{i}", completo=True, env={"detector": "owlvit", "dirty": False},
            raiz={"image_size": tamano, "rss_mb": rss, "model_load_ms": 11000.0 + i},
        )  # fmt: skip

    resumen = resumir_carpeta(tmp_path)

    assert resumen.detector == "owlvit" and resumen.dispositivo == "cpu"
    assert resumen.commit == "abc1234" and resumen.sucio is False
    assert resumen.tamano_imagen == "640x480"
    assert (resumen.repeticiones, resumen.calentamiento) == (5, 1)
    assert resumen.memoria_max_mb == 300.5  # el pico entre imágenes, no la mediana
    assert resumen.carga_modelo_ms == 11000.5  # la mediana
    assert resumen.faltantes == {} and resumen.avisos == ()


def test_model_load_ms_nulo_pero_presente_no_cuenta_como_dato_ausente(tmp_path, escribir_json):
    # Un detector sin modelo (whole) escribe "model_load_ms": null: no es un dato que falte.
    escribir_json(tmp_path, "a", completo=True)

    resumen = resumir_carpeta(tmp_path)

    assert resumen.carga_modelo_ms is None
    assert "model_load_ms" not in resumen.faltantes and resumen.faltantes == {}


def test_model_load_ms_que_no_esta_en_el_json_si_cuenta_como_ausente(tmp_path, escribir_json):
    escribir_json(tmp_path, "a", completo=True, sin=("model_load_ms",))

    assert resumir_carpeta(tmp_path).faltantes == {"model_load_ms": 1}


@pytest.mark.parametrize(
    ("clave", "nombre"),
    [("commit", "env.commit"), ("dirty", "env.dirty"), ("detector", "env.detector")],
)
def test_un_dato_de_env_a_null_cuenta_como_ausente(tmp_path, escribir_json, clave, nombre):
    escribir_json(tmp_path, "a", completo=True, env={clave: None})  # p. ej. sin git: commit null

    assert resumir_carpeta(tmp_path).faltantes == {nombre: 1}


def test_un_dato_presente_solo_en_algunos_json_se_cuenta_parcialmente(tmp_path, escribir_json):
    escribir_json(tmp_path, "a", completo=True, raiz={"rss_mb": 120.0})
    escribir_json(tmp_path, "b", completo=True, sin=("rss_mb",))
    escribir_json(tmp_path, "c", completo=True, raiz={"rss_mb": 150.0})

    resumen = resumir_carpeta(tmp_path)

    assert resumen.n_imagenes == 3 and resumen.memoria_max_mb == 150.0
    assert resumen.faltantes == {"rss_mb": 1}  # falta en 1 de los 3 JSON
    assert resumen.commit == "abc1234"


@pytest.mark.parametrize("donde", ["raiz", "env", "timing"])
def test_los_campos_numericos_se_buscan_en_el_nivel_superior_en_env_y_en_timing_ms(
    tmp_path, escribir_json, donde
):
    escribir_json(tmp_path, "a", **{donde: {"rss_mb": 256.0, "model_load_ms": 900.0}})

    resumen = resumir_carpeta(tmp_path)

    assert resumen.memoria_max_mb == 256.0 and resumen.carga_modelo_ms == 900.0


def test_repeticiones_y_calentamiento_iguales_se_muestran_tal_cual(tmp_path, escribir_json):
    for nombre in ("a", "b"):
        timing = {"repeticiones": 3, "calentamiento": 0}
        escribir_json(tmp_path, nombre, completo=True, timing=timing)

    resumen = resumir_carpeta(tmp_path)

    assert (resumen.repeticiones, resumen.calentamiento) == (3, 0)


def test_repeticiones_o_calentamiento_distintos_entre_json_se_resumen_como_varios(
    tmp_path, escribir_json
):
    escribir_json(tmp_path, "a", completo=True, timing={"repeticiones": 5, "calentamiento": 1})
    escribir_json(tmp_path, "b", completo=True, timing={"repeticiones": 1, "calentamiento": 1})

    resumen = resumir_carpeta(tmp_path)

    assert resumen.repeticiones == "varios" and resumen.calentamiento == 1


@pytest.mark.parametrize(
    ("tamano", "esperado"),
    [([640, 480], "640x480"), ([640.0, 480.0], "640x480"), ("640x480", "640x480")],
)
def test_el_tamano_de_imagen_se_muestra_en_el_orden_que_trae_el_json(
    tmp_path, escribir_json, tamano, esperado
):
    escribir_json(tmp_path, "a", raiz={"image_size": tamano})

    assert resumir_carpeta(tmp_path).tamano_imagen == esperado


def test_imagenes_de_distinto_tamano_se_resumen_como_varios(tmp_path, escribir_json):
    escribir_json(tmp_path, "a", raiz={"image_size": [640, 480]})
    escribir_json(tmp_path, "b", raiz={"image_size": [1024, 768]})

    assert resumir_carpeta(tmp_path).tamano_imagen == "varios"


def test_json_con_detectores_o_dispositivos_distintos_avisa_y_no_elige(tmp_path, escribir_json):
    escribir_json(tmp_path, "a", env={"detector": "whole", "device": "cpu"})
    escribir_json(tmp_path, "b", env={"detector": "owlvit", "device": "cuda"})

    resumen = resumir_carpeta(tmp_path)

    assert resumen.detector is None and resumen.dispositivo is None
    assert any("whole" in a and "owlvit" in a for a in resumen.avisos)
    assert any("cpu" in a and "cuda" in a for a in resumen.avisos)


def test_json_con_commits_distintos_avisa_y_lo_marca(tmp_path, escribir_json):
    escribir_json(tmp_path, "a", env={"commit": "abc1234"})
    escribir_json(tmp_path, "b", env={"commit": "def5678"})

    resumen = resumir_carpeta(tmp_path)

    assert resumen.commit is None and resumen.commits_distintos is True
    assert any("abc1234" in a and "def5678" in a for a in resumen.avisos)


def test_dirty_en_todos_los_json_marca_sucio(tmp_path, escribir_json):
    escribir_json(tmp_path, "a", completo=True, env={"dirty": True})
    escribir_json(tmp_path, "b", completo=True, env={"dirty": True})

    resumen = resumir_carpeta(tmp_path)

    assert resumen.sucio is True and resumen.alguno_sucio is True


def test_dirty_mixto_no_elige_valor_pero_si_recuerda_que_alguno_estaba_sucio(
    tmp_path, escribir_json
):
    escribir_json(tmp_path, "a", completo=True, env={"dirty": True})
    escribir_json(tmp_path, "b", completo=True, env={"dirty": False})

    resumen = resumir_carpeta(tmp_path)

    assert resumen.sucio is None and resumen.alguno_sucio is True
    assert any("dirty" in a for a in resumen.avisos)


def test_dirty_false_en_todos_no_marca_nada(tmp_path, escribir_json):
    escribir_json(tmp_path, "a", completo=True)

    resumen = resumir_carpeta(tmp_path)

    assert resumen.sucio is False and resumen.alguno_sucio is False


@pytest.mark.parametrize(
    ("donde", "clave", "valor", "nombre"),
    [
        ("raiz", "rss_mb", "mucho", "rss_mb"),
        ("raiz", "rss_mb", -5, "rss_mb"),
        ("raiz", "rss_mb", True, "rss_mb"),
        ("raiz", "model_load_ms", "rápido", "model_load_ms"),
        ("raiz", "image_size", [640, 480, 3], "image_size"),
        ("raiz", "image_size", 640, "image_size"),
        ("env", "dirty", "sí", "env.dirty"),
        ("env", "commit", 1234567, "env.commit"),
        ("env", "device", "", "env.device"),
        ("env", "detector", ["whole"], "env.detector"),
        ("timing", "repeticiones", "5", "timing_ms.repeticiones"),
        ("timing", "repeticiones", 0, "timing_ms.repeticiones"),
        ("timing", "calentamiento", -1, "timing_ms.calentamiento"),
        ("timing", "calentamiento", 1.5, "timing_ms.calentamiento"),
    ],
)
def test_un_campo_opcional_con_formato_incorrecto_se_ignora_y_se_avisa_sin_invalidar_el_json(
    tmp_path, escribir_json, donde, clave, valor, nombre
):
    escribir_json(tmp_path, "a", **{donde: {clave: valor}})

    resumen = resumir_carpeta(tmp_path)

    assert resumen.n_imagenes == 1 and resumen.ignorados == ()  # el JSON sigue valiendo
    assert all(getattr(resumen, campo) is None for campo in CAMPOS_OPCIONALES)
    assert any("a.json" in aviso and clave in aviso for aviso in resumen.avisos)
    assert nombre not in resumen.faltantes  # está, pero mal: se avisa aparte


# --- fila y tabla -----------------------------------------------------------------------


def test_fila_ejecucion_formatea_y_deja_vacio_lo_que_no_se_midio():
    fila = fila_ejecucion("runs/v0_whole", _resumen())

    assert list(fila) == list(COLUMNAS)
    assert fila["Carpeta"] == "runs/v0_whole"
    assert fila["Imágenes"] == "3" and fila["Mediana (ms)"] == "20.00"
    assert fila["Mediana detección (ms)"] == "10.00"
    columnas_opcionales = (
        "Detector", "Dispositivo", "Commit", "Sucio", "Tamaño imagen", "Repeticiones",
        "Calentamiento", "Carga modelo (ms)", "Memoria (MB)",
    )  # fmt: skip
    for columna in columnas_opcionales:
        assert fila[columna] == "", columna  # no medidas: vacías, no 0
    assert fila["Lechugas"] == "3" and fila["Sin detección"] == "1"
    assert (fila["ACCEPT"], fila["REVIEW"], fila["NON-CONFORMING"]) == ("1", "2", "0")
    assert fila["Observación"] == ""


def test_fila_ejecucion_con_todos_los_datos_del_json():
    resumen = _resumen(
        detector="owlvit", dispositivo="cpu", commit="abc1234", sucio=True,
        tamano_imagen="640x480", repeticiones=5, calentamiento=1, carga_modelo_ms=11030.04,
        memoria_max_mb=512.46,
    )  # fmt: skip

    fila = fila_ejecucion("runs/v0_owlvit", resumen)

    assert (fila["Detector"], fila["Dispositivo"]) == ("owlvit", "cpu")
    assert fila["Commit"] == "abc1234" and fila["Sucio"] == "sí"
    assert fila["Tamaño imagen"] == "640x480"
    assert (fila["Repeticiones"], fila["Calentamiento"]) == ("5", "1")
    assert fila["Carga modelo (ms)"] == "11030.0" and fila["Memoria (MB)"] == "512.5"
    assert fila_ejecucion("runs/x", _resumen(sucio=False))["Sucio"] == "no"
    assert fila_ejecucion("runs/x", _resumen(repeticiones="varios"))["Repeticiones"] == "varios"


def test_los_tiempos_de_menos_de_un_milisegundo_no_pierden_su_valor():
    resumen = _resumen(mediana_total_ms=0.0712, mediana_detect_ms=0.0456)

    fila = fila_ejecucion("runs/x", resumen)

    assert fila["Mediana (ms)"] == "0.07" and fila["Mediana detección (ms)"] == "0.05"


def test_renderizar_y_parsear_son_inversas_con_tildes_y_barras():
    con_barra = {**fila_ejecucion("runs/a", _resumen(detector="whole")), "Observación": "ñ | barra"}
    filas = [con_barra, fila_ejecucion("runs/b", _resumen(n_imagenes=5))]

    assert parsear_tabla(renderizar_tabla(filas)) == filas


_COLUMNAS_ANTIGUAS = (
    "Detector", "Dispositivo", "Imágenes", "Mediana (ms)", "Mediana detección (ms)",
    "Memoria (MB)", "Lechugas", "Sin detección", "ACCEPT", "REVIEW", "NON-CONFORMING",
    "Observación",
)  # fmt: skip
TABLA_ANTIGUA = "\n".join(
    [
        "| " + " | ".join(_COLUMNAS_ANTIGUAS) + " |",
        "| " + " | ".join("---" for _ in _COLUMNAS_ANTIGUAS) + " |",
        "| whole | CPU local | 2 | 0.07 | 0.02 |  | 2 | 0 | 0 | 2 | 0 | de antes |",
    ]
)


def test_una_tabla_con_las_columnas_antiguas_se_relee_y_se_completa_con_celdas_vacias():
    filas = parsear_tabla(TABLA_ANTIGUA)

    nueva = fila_ejecucion("runs/nueva", _resumen(detector="owlvit"))
    resultado = actualizar_filas(filas, nueva, observacion=None)
    releida = parsear_tabla(renderizar_tabla(resultado))

    assert [list(f) for f in releida] == [list(COLUMNAS)] * 2
    assert releida[0]["Detector"] == "whole" and releida[0]["Observación"] == "de antes"
    assert releida[0]["Imágenes"] == "2" and releida[0]["Mediana (ms)"] == "0.07"
    assert releida[0]["Carpeta"] == "" and releida[0]["Commit"] == ""  # sin dato, no 0


def test_parsear_un_texto_sin_tabla_devuelve_lista_vacia():
    assert parsear_tabla("Solo texto, sin tabla.") == []
    assert parsear_tabla("") == []


# --- la fila se identifica por su carpeta -----------------------------------------------


def test_actualizar_filas_anade_una_fila_nueva_sin_tocar_las_demas():
    existente = fila_ejecucion("runs/a", _resumen())
    nueva = fila_ejecucion("runs/b", _resumen(n_imagenes=7))

    filas = actualizar_filas([existente], nueva, observacion=None)

    assert filas == [existente, nueva]


def test_actualizar_filas_reemplaza_la_misma_carpeta_y_no_duplica():
    vieja = {**fila_ejecucion("runs/a", _resumen()), "Observación": "nota mía"}
    nueva = fila_ejecucion("runs/a", _resumen(n_imagenes=9))

    filas = actualizar_filas([vieja], nueva, observacion=None)

    assert len(filas) == 1
    assert filas[0]["Imágenes"] == "9"
    assert filas[0]["Observación"] == "nota mía"  # sin observación nueva, se conserva


def test_dos_carpetas_con_el_mismo_detector_y_dispositivo_son_dos_filas():
    resumen = _resumen(detector="whole", dispositivo="cpu")
    primera = fila_ejecucion("runs/a", resumen)

    filas = actualizar_filas([primera], fila_ejecucion("runs/b", resumen), observacion=None)

    assert [f["Carpeta"] for f in filas] == ["runs/a", "runs/b"]


def test_dos_carpetas_antiguas_sin_detector_ni_dispositivo_no_se_pisan():
    primera = fila_ejecucion("runs/v0_local", _resumen())  # sin detector ni dispositivo
    segunda = fila_ejecucion("runs/otra", _resumen(n_imagenes=2))

    filas = actualizar_filas([primera], segunda, observacion=None)

    assert [f["Carpeta"] for f in filas] == ["runs/v0_local", "runs/otra"]
    assert filas[0]["Imágenes"] == "3" and filas[1]["Imágenes"] == "2"


def test_una_fila_antigua_sin_carpeta_nunca_se_confunde_con_una_nueva():
    antigua = parsear_tabla(TABLA_ANTIGUA)  # su "Carpeta" está vacía
    nueva = fila_ejecucion("", _resumen())  # aunque llegara una carpeta vacía

    filas = actualizar_filas(antigua, nueva, observacion=None)

    assert len(filas) == 2 and filas[0]["Observación"] == "de antes"


def test_actualizar_filas_con_observacion_nueva_la_sustituye():
    vieja = {**fila_ejecucion("runs/a", _resumen()), "Observación": "vieja"}
    nueva = fila_ejecucion("runs/a", _resumen())

    filas = actualizar_filas([vieja], nueva, observacion="nueva")

    assert filas[0]["Observación"] == "nueva"


def test_actualizar_filas_no_modifica_la_lista_original():
    existente = fila_ejecucion("runs/a", _resumen())
    original = [dict(existente)]

    actualizar_filas(original, fila_ejecucion("runs/a", _resumen(n_imagenes=2)), None)

    assert original == [existente]


def test_el_contenido_del_bloque_explica_la_mediana_y_se_puede_releer():
    filas = [fila_ejecucion("runs/a", _resumen(detector="owlvit"))]

    contenido = contenido_bloque(filas)

    assert "timing_ms.total" in contenido and "calentamiento" in contenido
    assert parsear_tabla(contenido) == filas


def test_la_nota_de_la_tabla_ya_no_dice_que_no_hay_calentamiento():
    # Desde la tarea 5 el pipeline sí calienta y repite: aquello ya no es verdad.
    contenido = contenido_bloque([fila_ejecucion("runs/a", _resumen())])

    assert "sin calentamiento controlado" not in contenido
    assert "Repeticiones" in contenido and "tarea 5" in contenido
