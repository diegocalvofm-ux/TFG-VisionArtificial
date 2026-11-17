"""Métricas de la etapa 2: por grupo, solo con las muestras visibles, y línea base mayoritaria.

Los objetivos y las predicciones son matrices (filas, 6) de índices de etiqueta, con −1 para
NO_VISIBLE en los objetivos (ver ``datos.codificar``). Una fila NO_VISIBLE de un grupo no cuenta
en ese grupo, aunque el modelo prediga algo para ella.

Sin torch: estas funciones se prueban sin instalarlo.
"""

import copy

import numpy as np

from src.data.schema import GROUPS

GRUPOS = tuple(GROUPS)


def matriz_confusion(referencia: np.ndarray, prediccion: np.ndarray, n_clases: int) -> np.ndarray:
    """Matriz (n_clases, n_clases): fila = clase real, columna = clase predicha."""
    matriz = np.zeros((n_clases, n_clases), dtype=np.int64)
    np.add.at(matriz, (np.asarray(referencia, dtype=int), np.asarray(prediccion, dtype=int)), 1)
    return matriz


def exactitud_equilibrada(matriz: np.ndarray) -> float | None:
    """Media del recall de cada clase que aparece en la referencia (``None`` sin muestras).

    En lenguaje llano: para cada clase real, qué fracción de sus muestras acertó el modelo; y la
    media de esas fracciones. A diferencia de la exactitud normal, una clase rara pesa lo mismo
    que una frecuente: un modelo que siempre dice la mayoritaria saca 1/(número de clases).
    """
    reales = matriz.sum(axis=1)
    presentes = reales > 0
    if not presentes.any():
        return None
    return float(np.mean(np.diag(matriz)[presentes] / reales[presentes]))


def f1_macro(matriz: np.ndarray) -> float | None:
    """Media del F1 de las clases que aparecen en la referencia o en la predicción.

    En lenguaje llano: el F1 de una clase resume a la vez cuántas de sus muestras se encontraron y
    cuántas de las que se dijeron de esa clase lo eran (2·aciertos / (reales + predichas)). Una
    clase predicha que no existe en la referencia cuenta con F1 = 0 (es el criterio de sklearn).
    """
    reales, predichas = matriz.sum(axis=1), matriz.sum(axis=0)
    presentes = (reales + predichas) > 0
    if not presentes.any():
        return None
    f1 = 2 * np.diag(matriz)[presentes] / (reales + predichas)[presentes]
    return float(np.mean(f1))


def metricas_grupo(referencia: np.ndarray, prediccion: np.ndarray, etiquetas) -> dict:
    """Métricas de un grupo con solo las filas visibles (referencia >= 0).

    Devuelve ``n`` (filas visibles), ``exactitud_equilibrada``, ``f1_macro`` (``None`` si ``n`` es
    0), la ``matriz`` de confusión como listas, las ``etiquetas`` en el orden de la matriz y el
    ``soporte`` (filas reales de cada etiqueta).
    """
    referencia, prediccion = np.asarray(referencia), np.asarray(prediccion)
    visibles = referencia >= 0
    matriz = matriz_confusion(referencia[visibles], prediccion[visibles], len(etiquetas))
    return {
        "n": int(visibles.sum()),
        "exactitud_equilibrada": exactitud_equilibrada(matriz),
        "f1_macro": f1_macro(matriz),
        "matriz": matriz.tolist(),
        "etiquetas": list(etiquetas),
        "soporte": {e: int(n) for e, n in zip(etiquetas, matriz.sum(axis=1), strict=True)},
    }


def clases_mayoritarias(objetivos_train: np.ndarray) -> dict[str, str | None]:
    """La etiqueta más frecuente de cada grupo en train (``None`` si no hay ninguna visible).

    Es lo que diría el clasificador más tonto posible, "siempre la clase mayoritaria", que sirve
    de línea base: un modelo que no lo supera no ha aprendido nada útil. En un empate gana la
    etiqueta que va antes en ``schema.py``. Se calcula con train sin repeticiones.
    """
    salida = {}
    for j, (grupo, etiquetas) in enumerate(GROUPS.items()):
        columna = objetivos_train[:, j]
        cuentas = np.bincount(columna[columna >= 0], minlength=len(etiquetas))
        salida[grupo] = etiquetas[int(np.argmax(cuentas))] if cuentas.sum() > 0 else None
    return salida


def _grupo_con_linea_base(referencia, prediccion, grupo: str, mayoritaria: str | None) -> dict:
    etiquetas = GROUPS[grupo]
    modelo = metricas_grupo(referencia, prediccion, etiquetas)
    base = {"clase": mayoritaria, "exactitud_equilibrada": None, "f1_macro": None}
    if mayoritaria is not None:
        siempre = np.full(len(referencia), etiquetas.index(mayoritaria))
        medida = metricas_grupo(referencia, siempre, etiquetas)
        base |= {"exactitud_equilibrada": medida["exactitud_equilibrada"],
                 "f1_macro": medida["f1_macro"]}  # fmt: skip
    return {"modelo": modelo, "mayoritaria": base}


def criterio(grupos: dict) -> float | None:
    """Media de la exactitud equilibrada de los grupos con alguna muestra visible.

    Es el criterio con el que se elige el mejor punto del entrenamiento. Los grupos sin datos en
    el conjunto no entran (no hay nada que medir); sin ninguno, ``None``.
    """
    valores = [d["modelo"]["exactitud_equilibrada"] for d in grupos.values()
               if d["modelo"]["n"] > 0]  # fmt: skip
    return float(np.mean(valores)) if valores else None


def evaluar_conjunto(
    objetivos: np.ndarray,
    predicciones: np.ndarray,
    fuentes: list[str],
    mayoritarias: dict[str, str | None],
) -> dict:
    """Métricas de un conjunto (valid o test): por grupo y por fuente, frente a la línea base.

    En lenguaje llano: para cada grupo, las métricas del modelo con sus filas visibles y las del
    clasificador que siempre dice la clase mayoritaria de train; lo mismo separando las filas de
    cada fuente (propio, yssp, phs), porque una media global puede esconder que el modelo solo
    funciona con una de ellas. Además, el ``criterio`` (media de la exactitud equilibrada de los
    grupos con datos) y la lista ``grupos_con_datos``.

    Error: ``ValueError`` si las formas no coinciden.
    """
    objetivos, predicciones = np.asarray(objetivos), np.asarray(predicciones)
    if objetivos.shape != predicciones.shape or len(fuentes) != len(objetivos):
        raise ValueError(
            f"forma de objetivos {objetivos.shape}, de predicciones {predicciones.shape} y "
            f"{len(fuentes)} fuentes: deben coincidir"
        )
    fuentes = np.asarray(fuentes, dtype=object)

    def por_grupo(filas: np.ndarray) -> dict:
        return {grupo: _grupo_con_linea_base(objetivos[filas, j], predicciones[filas, j], grupo,
                                             mayoritarias.get(grupo))  # fmt: skip
                for j, grupo in enumerate(GRUPOS)}

    grupos = por_grupo(np.ones(len(objetivos), dtype=bool))
    return {
        "grupos": grupos,
        "por_fuente": {f: por_grupo(fuentes == f) for f in dict.fromkeys(fuentes.tolist())},
        "criterio": criterio(grupos),
        "grupos_con_datos": [g for g, d in grupos.items() if d["modelo"]["n"] > 0],
    }


def separar_matrices(evaluacion: dict) -> tuple[dict, dict]:
    """Quita las matrices de confusión de una evaluación y las devuelve aparte.

    El resumen queda legible y las matrices van a ``matrices_confusion.json`` con la forma
    ``{"grupos": {grupo: {"etiquetas", "matriz"}}, "por_fuente": {fuente: {grupo: ...}}}``. La
    evaluación de entrada no se modifica.
    """
    sin = copy.deepcopy(evaluacion)

    def extraer(grupos: dict) -> dict:
        matrices = {}
        for grupo, datos in grupos.items():
            modelo = datos["modelo"]
            matrices[grupo] = {"etiquetas": modelo["etiquetas"], "matriz": modelo.pop("matriz")}
        return matrices

    matrices = {
        "grupos": extraer(sin["grupos"]),
        "por_fuente": {f: extraer(g) for f, g in sin["por_fuente"].items()},
    }
    return sin, matrices
