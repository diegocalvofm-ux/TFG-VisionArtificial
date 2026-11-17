"""Pruebas de src/train/etapa2/perdida.py: pesos por clase y pérdida enmascarada por grupo.

Los números esperados se calculan a mano en cada prueba: la entropía cruzada de unos logits
``[a, b]`` con la clase 0 correcta es ``log(1 + e^(b - a))``.
"""

import math

import pytest

torch = pytest.importorskip("torch")

from src.train.etapa2.perdida import perdida_enmascarada, pesos_por_clase  # noqa: E402

LN2, LN3 = math.log(2), math.log(3)


# --- pesos por clase ----------------------------------------------------------------------


def test_pesos_inverso_de_la_raiz_con_numeros_conocidos():
    # 1/sqrt(100) = 0,1 y 1/sqrt(25) = 0,2; se normalizan para que el peso medio por muestra
    # sea 1: (100*0,1 + 25*0,2) = 15 sobre 125 muestras -> factor 125/15.
    assert pesos_por_clase([100, 25, 0], "inv_raiz") == pytest.approx([5 / 6, 5 / 3, 0.0])


def test_pesos_inverso_con_numeros_conocidos():
    # 1/100 y 1/25: (100*0,01 + 25*0,04) = 2 sobre 125 -> factor 62,5.
    assert pesos_por_clase([100, 25, 0], "inverso") == pytest.approx([0.625, 2.5, 0.0])


def test_sin_pesos_todas_las_clases_valen_1():
    assert pesos_por_clase([100, 25, 0], "ninguno") == [1.0, 1.0, 1.0]


@pytest.mark.parametrize("modo", ["inv_raiz", "inverso", "ninguno"])
def test_el_peso_medio_por_muestra_es_1(modo):
    recuentos = [7, 3, 1, 0]

    pesos = pesos_por_clase(recuentos, modo)

    assert sum(n * w for n, w in zip(recuentos, pesos, strict=True)) / 11 == pytest.approx(1.0)


def test_una_clase_sin_muestras_tiene_peso_0():
    assert pesos_por_clase([0, 4], "inv_raiz") == pytest.approx([0.0, 1.0])
    assert pesos_por_clase([0, 0], "inv_raiz") == [0.0, 0.0]


def test_un_modo_desconocido_da_error():
    with pytest.raises(ValueError, match="pesos_clase"):
        pesos_por_clase([1, 2], "logaritmico")


# --- pérdida enmascarada ------------------------------------------------------------------


def _salidas(**grupos):
    return {nombre: torch.tensor(valores, dtype=torch.float32, requires_grad=True)
            for nombre, valores in grupos.items()}  # fmt: skip


def test_con_logits_cero_la_perdida_es_el_logaritmo_del_numero_de_clases():
    salidas = _salidas(a=[[0, 0], [0, 0]], b=[[0, 0, 0], [0, 0, 0]])
    objetivos = torch.tensor([[0, 1], [1, 2]])

    total, por_grupo = perdida_enmascarada(salidas, objetivos)

    assert por_grupo == {"a": pytest.approx(LN2), "b": pytest.approx(LN3)}
    assert total.item() == pytest.approx((LN2 + LN3) / 2)


def test_un_grupo_no_visible_no_aporta_ni_recibe_gradiente():
    salidas = _salidas(a=[[0, 0], [0, 0]], b=[[5, -3, 1], [2, 2, 9]])
    objetivos = torch.tensor([[0, -1], [1, -1]])

    total, por_grupo = perdida_enmascarada(salidas, objetivos)
    total.backward()

    assert total.item() == pytest.approx(LN2) and por_grupo["b"] is None
    grad_b = salidas["b"].grad
    assert grad_b is None or torch.count_nonzero(grad_b) == 0


def test_cambiar_los_logits_de_un_grupo_no_visible_no_cambia_la_perdida():
    objetivos = torch.tensor([[1, -1], [0, -1]])
    uno, _ = perdida_enmascarada(_salidas(a=[[1, 2], [0, 3]], b=[[0, 0, 0], [0, 0, 0]]), objetivos)
    otro, _ = perdida_enmascarada(_salidas(a=[[1, 2], [0, 3]], b=[[9, -9, 4], [7, 1, -2]]),
                                  objetivos)  # fmt: skip

    assert uno.item() == pytest.approx(otro.item())


def test_solo_cuentan_las_filas_visibles_de_cada_grupo():
    # Fila 0: logits [2, 0], clase 0 -> log(1 + e^-2). La fila 1 es NO_VISIBLE.
    salidas = _salidas(a=[[2, 0], [0, 50]])
    objetivos = torch.tensor([[0], [-1]])

    total, _ = perdida_enmascarada(salidas, objetivos)

    assert total.item() == pytest.approx(math.log(1 + math.exp(-2)))


def test_el_total_es_la_media_de_los_grupos_con_alguna_muestra_visible():
    salidas = _salidas(a=[[0, 0]], b=[[0, 0, 0]], c=[[4, 1]])
    objetivos = torch.tensor([[1, 2, -1]])

    total, por_grupo = perdida_enmascarada(salidas, objetivos)

    assert por_grupo["c"] is None
    assert total.item() == pytest.approx((LN2 + LN3) / 2)  # c no entra en la media


def test_un_lote_sin_ninguna_muestra_visible_devuelve_none():
    salidas = _salidas(a=[[0, 0], [1, 1]], b=[[0, 0, 0], [1, 2, 3]])

    total, por_grupo = perdida_enmascarada(salidas, torch.full((2, 2), -1))

    assert total is None and por_grupo == {"a": None, "b": None}


def test_los_pesos_dan_la_media_ponderada_de_pytorch():
    # Logits [1, 0] en las dos filas: con la clase 0, log(1 + e^-1); con la 1, log(1 + e).
    # Pesos [1, 3]: (1 * l0 + 3 * l1) / (1 + 3).
    salidas = _salidas(a=[[1, 0], [1, 0]])
    objetivos = torch.tensor([[0], [1]])
    l0, l1 = math.log(1 + math.exp(-1)), math.log(1 + math.e)

    total, _ = perdida_enmascarada(salidas, objetivos, {"a": torch.tensor([1.0, 3.0])})

    assert total.item() == pytest.approx((l0 + 3 * l1) / 4)


def test_un_grupo_cuyas_muestras_tienen_peso_0_no_aporta():
    salidas = _salidas(a=[[0, 0]], b=[[0, 0, 0]])
    objetivos = torch.tensor([[0, 1]])

    total, por_grupo = perdida_enmascarada(
        salidas, objetivos, {"a": torch.tensor([0.0, 1.0]), "b": torch.tensor([1.0, 1.0, 1.0])}
    )

    assert por_grupo["a"] is None and total.item() == pytest.approx(LN3)


def test_las_columnas_de_los_objetivos_deben_coincidir_con_los_grupos():
    with pytest.raises(ValueError, match="columnas"):
        perdida_enmascarada(_salidas(a=[[0, 0]]), torch.tensor([[0, 1]]))
