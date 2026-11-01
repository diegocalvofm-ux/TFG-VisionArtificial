"""Lectura de las métricas de valid que escribe RF-DETR (``metrics.csv`` de Lightning).

``rfdetr.train()`` no devuelve métricas: el ``CSVLogger`` de PyTorch Lightning las va guardando en
``<salida>/metrics.csv``, con una fila por paso de entrenamiento (columnas ``train/*``) y una fila
de validación por época (columnas ``val/*``), dejando en blanco las celdas que no tocan.
"""

import csv
import math
from pathlib import Path

# Columnas de rfdetr 1.11.2: AP con IoU 0.50 y AP medio con IoU de 0.50 a 0.95.
COLUMNA_AP50 = "val/mAP_50"
COLUMNA_AP50_95 = "val/mAP_50_95"


def _numero(texto) -> float | None:
    """Un número finito, o ``None`` si la celda está vacía, es ``nan`` o no es un número."""
    if texto is None or not str(texto).strip():
        return None
    try:
        valor = float(texto)
    except ValueError:
        return None
    return valor if math.isfinite(valor) else None


def leer_metricas_csv(ruta: str | Path) -> list[dict]:
    """Lee las métricas de valid de cada época de un ``metrics.csv`` de Lightning.

    En lenguaje llano: recorre el CSV y, para cada época, junta los valores de las columnas
    ``val/*`` (si una época repite una métrica, se queda con el último valor). Devuelve una lista
    ordenada por época con ``epoca`` (cuenta desde 1: la época de Lightning más uno),
    ``epoca_csv`` (el número tal como viene en el CSV, desde 0), ``AP50``, ``AP50_95`` (``None``
    si faltan; nunca se inventan) y ``val`` con todas las métricas de valid de esa época.

    Errores: ``FileNotFoundError`` si no existe; ``ValueError`` si no hay columna ``epoch`` o si
    ninguna época trae métricas de valid.
    """
    ruta = Path(ruta)
    if not ruta.is_file():
        raise FileNotFoundError(f"No existe {ruta.name}: el entrenamiento no llegó a escribirlo")

    with ruta.open(encoding="utf-8", newline="") as archivo:
        lector = csv.DictReader(archivo)
        columnas = lector.fieldnames or []
        if "epoch" not in columnas:
            raise ValueError(f"{ruta.name} no tiene la columna 'epoch'")
        de_valid = [c for c in columnas if c.startswith("val/")]
        por_epoca: dict[int, dict[str, float]] = {}
        for fila in lector:
            epoca = _numero(fila.get("epoch"))
            if epoca is None:
                continue
            for columna in de_valid:
                valor = _numero(fila.get(columna))
                if valor is not None:
                    por_epoca.setdefault(int(epoca), {})[columna] = valor

    if not por_epoca:
        raise ValueError(
            f"{ruta.name} no tiene métricas de valid (columnas val/*) en ninguna época"
        )
    return [
        {
            "epoca": epoca + 1,
            "epoca_csv": epoca,
            "AP50": valores.get(COLUMNA_AP50),
            "AP50_95": valores.get(COLUMNA_AP50_95),
            "val": dict(valores),
        }
        for epoca, valores in sorted(por_epoca.items())
    ]


def mejor_punto(metricas: list[dict]) -> dict | None:
    """La época con mayor AP50:95 (el criterio con el que rfdetr elige su mejor checkpoint).

    En un empate gana la primera época. Devuelve ``epoca``, ``epoca_csv``, ``AP50`` y
    ``AP50_95``, o ``None`` si ninguna época tiene AP50:95.
    """
    candidatas = [m for m in metricas if m.get("AP50_95") is not None]
    if not candidatas:
        return None
    mejor = max(candidatas, key=lambda m: (m["AP50_95"], -m["epoca_csv"]))
    return {
        "epoca": mejor["epoca"],
        "epoca_csv": mejor["epoca_csv"],
        "AP50": mejor["AP50"],
        "AP50_95": mejor["AP50_95"],
    }
