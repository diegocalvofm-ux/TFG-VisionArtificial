"""Pruebas de src/data/recortes.py: geometría de los recortes y huellas perceptuales (dHash)."""

import random

import cv2
import numpy as np
import pytest

from src.data import recortes
from src.data.recortes import (
    agrupar,
    ampliar_caja,
    contar_bits,
    dhash,
    huellas,
    motivo_caja_rara,
    pares_cercanos,
    redimensionar_lado_mayor,
    se_sale_en_parte,
)


def _imagen_suave(semilla: int, ancho: int = 96, alto: int = 72) -> np.ndarray:
    """Manchas de color al azar ampliadas: una imagen con estructura distinta por semilla."""
    pequena = np.random.default_rng(semilla).integers(0, 256, (6, 8, 3), dtype=np.uint8)
    return cv2.resize(pequena, (ancho, alto), interpolation=cv2.INTER_CUBIC)


# --- Margen y recorte a los bordes ---------------------------------------------------------


def test_margen_del_10_por_ciento_a_cada_lado():
    # Caja de 20x30 en (10, 20): se añaden 2 px a izquierda y derecha y 3 px arriba y abajo.
    assert ampliar_caja((10, 20, 20, 30), 0.10, 100, 100) == (8, 17, 32, 53)


def test_margen_cero_deja_la_caja_igual():
    assert ampliar_caja((10, 20, 20, 30), 0.0, 100, 100) == (10, 20, 30, 50)


def test_margen_redondea_hacia_fuera_sin_ruido_de_coma_flotante():
    # 0.1 * 30 = 3.0000000000000004 en coma flotante: no debe dar un píxel de más.
    assert ampliar_caja((10, 10, 30, 30), 0.10, 100, 100) == (7, 7, 43, 43)
    # Con decimales de verdad se redondea hacia fuera (la caja nunca se encoge).
    assert ampliar_caja((10.5, 10.5, 10, 10), 0.0, 100, 100) == (10, 10, 21, 21)


def test_recorte_a_los_bordes_en_una_esquina():
    # Caja pegada a la esquina superior izquierda: el margen se corta en 0.
    assert ampliar_caja((0, 0, 50, 40), 0.10, 60, 45) == (0, 0, 55, 44)
    # Pegada a la inferior derecha: se corta en el ancho y el alto de la imagen.
    assert ampliar_caja((30, 20, 30, 25), 0.20, 60, 45) == (24, 15, 60, 45)


def test_caja_de_imagen_entera_con_margen_es_la_imagen_entera():
    assert ampliar_caja((0, 0, 64, 48), 0.10, 64, 48) == (0, 0, 64, 48)


# --- Cajas raras ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("caja", "motivo"),
    [
        ((10, 10, 0, 20), "degenerada"),
        ((10, 10, 20, -1), "degenerada"),
        ((200, 10, 20, 20), "fuera de la imagen"),
        ((-30, 10, 20, 20), "fuera de la imagen"),
        ((10, 10, 31, 40), "pequeña"),
        ((10, 10, 40, 31.5), "pequeña"),
        ((10, 10, 32, 32), None),
    ],
)
def test_motivo_caja_rara(caja, motivo):
    assert motivo_caja_rara(caja, 100, 100, min_lado_px=32) == motivo


def test_caja_que_se_sale_en_parte_no_es_rara_pero_se_detecta():
    caja = (90, 10, 20, 40)  # llega a x = 110 en una imagen de 100 de ancho
    assert motivo_caja_rara(caja, 100, 100, min_lado_px=8) is None
    assert se_sale_en_parte(caja, 100, 100)
    assert not se_sale_en_parte((10, 10, 20, 20), 100, 100)
    assert not se_sale_en_parte((0, 0, 100, 100), 100, 100)  # justo en el borde: no se sale


# --- Redimensionado sin deformar -----------------------------------------------------------


@pytest.mark.parametrize(
    ("alto", "ancho", "esperado"),
    [
        (50, 100, (256, 512)),  # apaisada, ampliando
        (100, 50, (512, 256)),  # vertical, ampliando
        (1000, 2000, (256, 512)),  # apaisada, reduciendo
        (333, 1000, (170, 512)),  # 333 * 0,512 = 170,496 -> 170
        (512, 512, (512, 512)),  # ya tiene el tamaño
    ],
)
def test_redimensionar_lado_mayor_sin_deformar(alto, ancho, esperado):
    imagen = np.zeros((alto, ancho, 3), dtype=np.uint8)
    salida = redimensionar_lado_mayor(imagen, 512)
    assert salida.shape == (*esperado, 3)
    assert max(salida.shape[:2]) == 512
    # Sin deformar: los dos lados se escalan por el mismo factor (salvo el redondeo a píxeles).
    escala = 512 / max(alto, ancho)
    assert abs(salida.shape[0] - alto * escala) <= 0.5 + 1e-9
    assert abs(salida.shape[1] - ancho * escala) <= 0.5 + 1e-9


def test_redimensionar_no_modifica_la_original():
    imagen = _imagen_suave(1)
    copia = imagen.copy()
    redimensionar_lado_mayor(imagen, 512)
    np.testing.assert_array_equal(imagen, copia)


# --- dHash ---------------------------------------------------------------------------------


def test_dhash_de_imagen_constante_es_cero():
    assert dhash(np.full((40, 50, 3), 128, dtype=np.uint8)) == 0


def test_dhash_de_degradados_conocidos():
    creciente = np.tile(np.linspace(0, 255, 90).astype(np.uint8), (80, 1))
    assert dhash(creciente) == 2**64 - 1  # cada píxel es más claro que el de su izquierda
    assert dhash(creciente[:, ::-1]) == 0  # y al revés, ninguno


def test_dhash_tiene_64_bits_y_es_estable():
    imagen = _imagen_suave(3)
    assert 0 <= dhash(imagen) < 2**64
    assert dhash(imagen) == dhash(imagen.copy())


def test_dhash_cambia_poco_con_un_cambio_leve_de_brillo():
    imagen = _imagen_suave(4)
    mas_clara = cv2.add(imagen, np.full_like(imagen, 6))
    assert (dhash(imagen) ^ dhash(mas_clara)).bit_count() <= 4


def test_dhash_de_imagenes_distintas_esta_lejos():
    assert (dhash(_imagen_suave(5)) ^ dhash(_imagen_suave(6))).bit_count() > 10


def test_huellas_sin_y_con_giros_y_espejos():
    imagen = _imagen_suave(7)
    assert huellas(imagen, giros_y_espejos=False) == (dhash(imagen),)
    variantes = huellas(imagen, giros_y_espejos=True)
    assert len(variantes) == 8
    assert variantes[0] == dhash(imagen)  # la primera es siempre la directa
    assert dhash(np.fliplr(imagen)) in variantes
    assert dhash(np.rot90(imagen)) in variantes


def test_un_espejo_solo_se_detecta_con_giros_y_espejos():
    imagen = _imagen_suave(8)
    espejo = np.ascontiguousarray(np.fliplr(imagen))
    sin = pares_cercanos([huellas(imagen, False), huellas(espejo, False)], umbral=4)
    con = pares_cercanos([huellas(imagen, True), huellas(espejo, True)], umbral=4)
    assert sin == []
    assert con == [(0, 1, 0)]


# --- Distancias y grupos -------------------------------------------------------------------


def test_contar_bits_coincide_con_bit_count_por_las_dos_vias():
    azar = random.Random(0)
    valores = [0, 1, 2**64 - 1, 2**63] + [azar.getrandbits(64) for _ in range(200)]
    arreglo = np.array(valores, dtype=np.uint64)
    esperado = [v.bit_count() for v in valores]
    assert contar_bits(arreglo, con_tabla=True).tolist() == esperado
    if hasattr(np, "bitwise_count"):
        assert contar_bits(arreglo, con_tabla=False).tolist() == esperado


@pytest.mark.parametrize("sin_bitwise_count", [False, True])
def test_pares_cercanos_coincide_con_la_fuerza_bruta(monkeypatch, sin_bitwise_count):
    if sin_bitwise_count:
        monkeypatch.setattr(recortes, "_HAY_BITWISE_COUNT", False)
    azar = random.Random(1)
    base = [azar.getrandbits(64) for _ in range(40)]
    # Copias con 0 a 6 bits cambiados: unas caen dentro del umbral y otras no.
    copias = [b ^ sum(1 << azar.randrange(64) for _ in range(azar.randrange(7))) for b in base[:20]]
    todas = [(h,) for h in base + copias]
    esperado = sorted(
        (i, j, (todas[i][0] ^ todas[j][0]).bit_count())
        for i in range(len(todas))
        for j in range(i + 1, len(todas))
        if (todas[i][0] ^ todas[j][0]).bit_count() <= 4
    )
    assert pares_cercanos(todas, umbral=4, bloque=7) == esperado
    assert esperado  # la prueba tiene pares que encontrar


def test_pares_cercanos_mira_las_variantes_en_las_dos_direcciones():
    # La variante de 0 coincide con la directa de 1, pero no al revés: el par se encuentra igual.
    assert pares_cercanos([(10, 99), (99, 12345)], umbral=0) == [(0, 1, 0)]


def test_agrupar_une_cadenas_y_deja_fuera_los_sueltos():
    pares = [(0, 2, 1), (2, 5, 3), (3, 4, 0)]
    assert agrupar(7, pares) == [[0, 2, 5], [3, 4]]
    assert agrupar(3, []) == []
