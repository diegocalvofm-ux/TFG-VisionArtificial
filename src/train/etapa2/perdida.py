"""Pesos por clase y pérdida de la etapa 2: una entropía cruzada por grupo, solo con lo visible.

Cada recorte tiene seis etiquetas, pero muchas son NO_VISIBLE (los recortes públicos solo traen
``daño_bio``; en la vista de base no se ve el color...). NO_VISIBLE no es una clase que el modelo
deba predecir sino la falta de etiqueta: esa casilla no aporta nada a la pérdida.
"""

import math

import torch
import torch.nn.functional as F

from src.train.etapa2.config import MODOS_PESOS


def pesos_por_clase(recuentos: list[int], modo: str) -> list[float]:
    """Peso de cada clase de un grupo a partir de cuántas muestras tiene en train.

    En lenguaje llano: con ``inv_raiz`` una clase con ``n`` muestras pesa ``1/√n`` y con
    ``inverso``, ``1/n``; así las clases raras cuentan más y el modelo no se limita a decir
    siempre la mayoritaria (``inv_raiz`` compensa menos, para no exagerar con clases de muy pocas
    muestras). Después se reescalan para que el peso medio por muestra sea 1 (la pérdida conserva
    su escala). Una clase sin muestras recibe 0: nunca es el objetivo de una muestra de train.
    Con ``ninguno`` todas pesan 1.

    Error: ``ValueError`` si ``modo`` no es ninguno, inv_raiz ni inverso.
    """
    if modo not in MODOS_PESOS:
        raise ValueError(f"perdida.pesos_clase: modo {modo!r} desconocido (válidos: "
                         f"{', '.join(MODOS_PESOS)})")  # fmt: skip
    if modo == "ninguno":
        return [1.0] * len(recuentos)
    potencia = 0.5 if modo == "inv_raiz" else 1.0
    crudos = [n**-potencia if n > 0 else 0.0 for n in recuentos]
    total = sum(recuentos)
    ponderado = sum(n * w for n, w in zip(recuentos, crudos, strict=True))
    if ponderado == 0:
        return [0.0] * len(recuentos)
    return [w * total / ponderado for w in crudos]


def perdida_enmascarada(
    salidas: dict[str, torch.Tensor],
    objetivos: torch.Tensor,
    pesos: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor | None, dict[str, float | None]]:
    """Pérdida total de un lote y la de cada grupo, sin contar las casillas NO_VISIBLE.

    En lenguaje llano:

    - ``salidas`` son los logits de cada grupo (lote, clases del grupo) y ``objetivos`` la matriz
      (lote, grupos) de ``codificar``: la columna ``j`` es el ``j``-ésimo grupo de ``salidas`` y
      −1 es NO_VISIBLE.
    - En cada grupo se calcula la entropía cruzada solo con las filas visibles. Con ``pesos`` es
      la media ponderada de PyTorch: Σ peso(clase)·pérdida / Σ peso(clase).
    - El total es la media de los grupos que tienen alguna fila visible en el lote (con peso
      mayor que 0): un grupo sin datos en el lote no tira de la media hacia cero.
    - Si ningún grupo tiene filas visibles, el total es ``None`` (ese lote no se usa).

    Devuelve ``(total, {grupo: pérdida o None})``.
    Error: ``ValueError`` si las columnas de ``objetivos`` no son tantas como los grupos.
    """
    if objetivos.ndim != 2 or objetivos.shape[1] != len(salidas):
        raise ValueError(
            f"objetivos tiene {tuple(objetivos.shape)} y hay {len(salidas)} grupos: hacen falta "
            "tantas columnas como grupos"
        )
    por_grupo: dict[str, float | None] = {}
    validas = []
    for j, (grupo, logits) in enumerate(salidas.items()):
        objetivo = objetivos[:, j]
        visibles = objetivo >= 0
        peso = None if pesos is None else pesos[grupo].to(logits.device, logits.dtype)
        if visibles.any() and (peso is None or peso[objetivo[visibles]].sum() > 0):
            valor = F.cross_entropy(logits[visibles], objetivo[visibles], weight=peso)
            validas.append(valor)
            por_grupo[grupo] = float(valor.detach())
        else:
            por_grupo[grupo] = None
    if not validas:
        return None, por_grupo
    total = torch.stack(validas).mean()
    if not math.isfinite(float(total.detach())):
        raise RuntimeError(f"la pérdida no es finita ({float(total.detach())}): revisa las tasas")
    return total, por_grupo
