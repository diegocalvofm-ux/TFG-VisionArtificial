"""Pruebas de src/utils/profiling.py: mediana de tiempos, calentamiento y memoria residente."""

import inspect
import time
from types import SimpleNamespace

import pytest

from src.utils import profiling
from src.utils.profiling import mediana, medir, rss_mb

MIB = 1024**2


# --- Caso 1: la mediana ---------------------------------------------------------------------


def test_caso1_mediana_con_numero_impar_de_repeticiones(reloj, con_duraciones):
    func = con_duraciones(reloj, [10, 30, 20])

    resultado = medir(func, repeticiones=3, calentamiento=0, reloj=reloj)

    assert resultado == pytest.approx(20)  # el del medio, no la media (20 también) ni el máximo


def test_caso1_mediana_con_numero_par_de_repeticiones_promedia_las_dos_centrales(
    reloj, con_duraciones
):
    func = con_duraciones(reloj, [10, 20, 30, 40])

    resultado = medir(func, repeticiones=4, calentamiento=0, reloj=reloj)

    assert resultado == pytest.approx(25)


def test_la_mediana_no_se_deja_enganar_por_un_valor_extremo(reloj, con_duraciones):
    func = con_duraciones(reloj, [10, 12, 11, 5000, 13])  # la media sería 1009

    assert medir(func, repeticiones=5, calentamiento=0, reloj=reloj) == pytest.approx(12)


def test_el_resultado_esta_en_milisegundos(reloj, con_duraciones):
    func = con_duraciones(reloj, [10])

    assert medir(func, repeticiones=1, calentamiento=0, reloj=reloj) == pytest.approx(10)


# --- Caso 2: el calentamiento no cuenta --------------------------------------------------------


def test_caso2_el_calentamiento_no_cuenta_en_la_mediana(reloj, con_duraciones):
    func = con_duraciones(reloj, [500, 10, 30, 20])  # la primera es el arranque en frío

    resultado = medir(func, repeticiones=3, calentamiento=1, reloj=reloj)

    assert resultado == pytest.approx(20)  # si contase el calentamiento saldría 25


def test_sin_calentamiento_la_primera_llamada_si_cuenta(reloj, con_duraciones):
    func = con_duraciones(reloj, [500, 10, 30, 20])

    resultado = medir(func, repeticiones=4, calentamiento=0, reloj=reloj)

    assert resultado == pytest.approx(25)  # mediana de [10, 20, 30, 500]


def test_el_calentamiento_se_ejecuta_pero_no_se_mide(reloj, con_duraciones):
    func = con_duraciones(reloj, [500, 600, 10, 30, 20])

    medir(func, repeticiones=3, calentamiento=2, reloj=reloj)

    assert len(func.llamadas) == 5  # 2 de calentamiento + 3 medidas
    assert reloj.llamadas == 6  # el reloj solo se mira alrededor de las 3 medidas


def test_con_los_valores_por_defecto_se_ejecuta_seis_veces_y_devuelve_un_numero():
    llamadas = []

    resultado = medir(lambda: llamadas.append(1))

    assert len(llamadas) == 6  # 1 de calentamiento + 5 repeticiones
    assert isinstance(resultado, float) and resultado >= 0


def test_con_una_repeticion_y_sin_calentamiento_se_ejecuta_una_sola_vez():
    llamadas = []

    medir(lambda: llamadas.append(1), repeticiones=1, calentamiento=0)

    assert len(llamadas) == 1


def test_el_reloj_por_defecto_es_perf_counter():
    assert inspect.signature(medir).parameters["reloj"].default is time.perf_counter


@pytest.mark.parametrize(
    ("repeticiones", "calentamiento"), [(0, 1), (-1, 1), (2.5, 1), (True, 1), (5, -1), (5, 1.5)]
)
def test_valores_invalidos_lanzan_value_error_sin_ejecutar_nada(repeticiones, calentamiento):
    llamadas = []

    with pytest.raises(ValueError, match="repeticiones|calentamiento"):
        medir(lambda: llamadas.append(1), repeticiones=repeticiones, calentamiento=calentamiento)

    assert llamadas == []


def test_si_la_funcion_falla_el_error_se_propaga():
    def rota():
        raise RuntimeError("fallo del detector")

    with pytest.raises(RuntimeError, match="fallo del detector"):
        medir(rota, repeticiones=2, calentamiento=0)


# --- mediana --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("valores", "esperado"),
    [([3, 1, 2], 2), ([4, 1, 3, 2], 2.5), ([7], 7), ([5.5, 1.5], 3.5), ([9, 1, 5, 5], 5)],
)
def test_mediana(valores, esperado):
    resultado = mediana(valores)

    assert resultado == pytest.approx(esperado) and isinstance(resultado, float)


def test_mediana_de_una_lista_vacia_lanza_value_error():
    with pytest.raises(ValueError, match="vacía"):
        mediana([])


# --- Caso 10: memoria residente ---------------------------------------------------------------


def _psutil_falso(*valores_rss):
    """Un psutil de mentira cuyo proceso va devolviendo, en orden, estos bytes residentes."""
    pendientes = iter(valores_rss)

    def memory_info():
        return SimpleNamespace(rss=next(pendientes))

    return SimpleNamespace(Process=lambda: SimpleNamespace(memory_info=memory_info))


def test_caso10_rss_mb_real_devuelve_un_numero_positivo():
    resultado = rss_mb()

    assert isinstance(resultado, float) and resultado > 0


def test_rss_mb_convierte_bytes_a_mb(monkeypatch):
    monkeypatch.setattr(profiling, "psutil", _psutil_falso(100 * MIB))

    assert rss_mb() == pytest.approx(100.0)


def test_rss_mb_es_la_memoria_actual_y_no_el_pico(monkeypatch):
    monkeypatch.setattr(profiling, "psutil", _psutil_falso(200 * MIB, 100 * MIB))

    primera, segunda = rss_mb(), rss_mb()

    assert primera == pytest.approx(200) and segunda == pytest.approx(100)  # baja: no es un máximo


def test_el_docstring_de_rss_mb_aclara_que_no_es_el_pico():
    documentacion = rss_mb.__doc__

    assert "residente" in documentacion and "no el pico" in documentacion
