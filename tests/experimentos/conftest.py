"""Fixtures de las pruebas del registro de experimentos (datos sintéticos, nada real)."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

# Copia de la estructura de docs/plantillas/experimento.md, con los marcadores de Obsidian.
PLANTILLA = """---
id: "{{title}}"
fecha: {{date:YYYY-MM-DD}}
estado: planificado
objetivo:
dataset:
dataset_version:
modelo:
config:
commit:
entorno:
python:
torch:
metrica_principal:
valor:
carpeta_drive:
tags: [experimento]
---

# {{title}}

## Objetivo e hipótesis

## Datos

## Configuración

## Resultados

## Análisis de errores

## Conclusión y siguiente paso
"""

# Lo que escribe el pipeline desde la tarea 5 (src.pipeline.run.procesar_imagen), además de lo
# de siempre: las ocho claves de env, image_size, model_load_ms, rss_mb y, dentro de
# timing_ms, repeticiones y calentamiento. model_load_ms es null con el detector whole.
ENV_COMPLETO = {
    "torch": "2.14.1",
    "sistema": "Windows 11",
    "device": "cpu",
    "detector": "whole",
    "commit": "abc1234",
    "dirty": False,
}
RAIZ_COMPLETA = {"image_size": [640, 480], "rss_mb": 80.0, "model_load_ms": None}
TIMING_COMPLETO = {"repeticiones": 5, "calentamiento": 1}


@pytest.fixture
def docs(tmp_path):
    """Una bóveda de Obsidian mínima: docs/plantillas/experimento.md y docs/experimentos/."""
    raiz = tmp_path / "docs"
    plantilla = raiz / "plantillas" / "experimento.md"
    plantilla.parent.mkdir(parents=True)
    plantilla.write_text(PLANTILLA, encoding="utf-8")
    carpeta = raiz / "experimentos"
    carpeta.mkdir()
    return SimpleNamespace(raiz=raiz, plantilla=plantilla, carpeta=carpeta)


@pytest.fixture
def escribir_json():
    """Devuelve una función que escribe un JSON como los del pipeline.

    Por defecto escribe un JSON **antiguo** (anterior a la tarea 5: solo ``python`` y
    ``opencv`` en ``env`` y ``detect``, ``assess`` y ``total`` en ``timing_ms``). Con
    ``completo=True`` añade todo lo que escribe hoy el pipeline. ``env=``, ``raiz=`` (nivel
    superior del JSON) y ``timing=`` (dentro de ``timing_ms``) añaden o cambian campos, y
    ``sin=("clave", ...)`` quita campos de cualquiera de los tres sitios.
    """

    def escribir(
        carpeta: Path,
        nombre: str,
        *,
        detect_ms=1.0,
        total_ms=2.0,
        niveles=("REVIEW",),
        completo=False,
        env=None,
        raiz=None,
        timing=None,
        sin=(),
    ) -> Path:
        carpeta.mkdir(parents=True, exist_ok=True)
        datos = {
            "image": f"{nombre}.jpg",
            "lettuces": [
                {
                    "box": [0, 0, 10, 10],
                    "score": 1.0,
                    "groups": {},
                    "global": nivel,
                    "provisional_labels": [],
                }
                for nivel in niveles
            ],
            "timing_ms": {
                "detect": detect_ms,
                "assess": 0.0,
                "total": total_ms,
                **(TIMING_COMPLETO if completo else {}),
                **(timing or {}),
            },
            "env": {
                "python": "3.12.10",
                "opencv": "5.0.0",
                **(ENV_COMPLETO if completo else {}),
                **(env or {}),
            },
            **(RAIZ_COMPLETA if completo else {}),
            **(raiz or {}),
        }
        for clave in sin:
            for sitio in (datos, datos["env"], datos["timing_ms"]):
                sitio.pop(clave, None)
        ruta = carpeta / f"{nombre}.json"
        ruta.write_text(json.dumps(datos, ensure_ascii=False), encoding="utf-8")
        return ruta

    return escribir
