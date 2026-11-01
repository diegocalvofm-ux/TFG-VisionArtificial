"""Entrenamiento de RF-DETR para la etapa 1 (localizar la lechuga) y registro de la ejecución.

Uso (desde la raíz del repositorio):
    python -m src.train.entrenar --exp-id mi_prueba --smoke
    python -m src.train.entrenar --exp-id etapa1_rfdetr_n_001
(``--config`` vale ``configs/train/etapa1_rfdetr_n.yaml`` por defecto.)

``--smoke`` entrena 1 época con unas 20 imágenes de train y todo valid (una prueba rápida de que
todo funciona). Cada ejecución crea ``runs/<exp_id>/`` con la configuración usada, el entorno
(versiones, GPU, commit y estado de git), el informe del dataset, las métricas de valid por época,
el mejor punto, el tiempo de entrenamiento y los pesos. El dataset de trabajo (con las repeticiones
de svodk) NO va ahí: se crea aparte (``datos.trabajo`` o ``--trabajo``).

Solo ``entrenar_rfdetr`` importa ``rfdetr`` (dentro de la función): el resto, y las pruebas, no lo
necesitan. Todo parámetro sale del YAML; ver ``src/train/config.py``.
"""

import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path

import yaml

from src.experimentos.privacidad import mostrar, sanear
from src.train.config import ConfigEntrenamiento, cargar_config_entrenamiento
from src.train.datos import crear_copia_de_trabajo
from src.train.entorno import entorno_entrenamiento
from src.train.metricas import leer_metricas_csv, mejor_punto

RAIZ = Path(__file__).resolve().parents[2]
CONFIG_POR_DEFECTO = RAIZ / "configs" / "train" / "etapa1_rfdetr_n.yaml"

# Nombre de la métrica con la que se registra el experimento (rfdetr elige su mejor checkpoint por
# este mismo criterio: AP medio con IoU de 0.50 a 0.95).
METRICA_PRINCIPAL = "AP50_95"
# Pesos que se mueven a runs/<exp_id>/pesos/ (rfdetr deja el resto en runs/<exp_id>/rfdetr/).
PESOS_A_GUARDAR = ("checkpoint_best_ema.pth", "checkpoint_best_total.pth")

_ID_EJECUCION = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


def _parametros(config: ConfigEntrenamiento, smoke: bool) -> dict:
    """Lo que recibe el entrenador: los valores del YAML (con las épocas del smoke si toca)."""
    return {
        "clase": config.modelo_clase,
        "resolucion": config.resolucion,
        "clases": list(config.clases),
        "epocas": config.smoke_epocas if smoke else config.epocas,
        "batch_size": config.batch_size,
        "grad_accum_steps": config.grad_accum_steps,
        "lr": config.lr,
        "lr_encoder": config.lr_encoder,
        "amp_dtype": config.amp_dtype,
        "num_workers": config.num_workers,
        "early_stopping": config.early_stopping,
        "checkpoint_interval": config.checkpoint_interval,
        "tensorboard": config.tensorboard,
        "dispositivo": config.dispositivo,
        "semilla": config.semilla,
    }


def entrenar_rfdetr(parametros: dict, dataset_dir: Path, salida: Path) -> None:
    """Entrena un modelo de RF-DETR con ``rfdetr`` (única función que lo importa).

    En lenguaje llano: crea el modelo por el nombre de su clase (con los pesos COCO que trae por
    defecto) y llama a ``train()`` con los parámetros del YAML. Los nombres de los argumentos son
    los de ``rfdetr.config.TrainConfig`` 1.11.2, que rechaza los que no existen. ``train()`` no
    devuelve métricas: las escribe en ``<salida>/metrics.csv``.
    """
    try:
        import rfdetr
    except ImportError as error:
        raise ImportError(
            "No está instalado rfdetr. Instálalo con `pip install -c <constraints> -r "
            "requirements-train.txt` (ver notebooks/02_entrenamiento_etapa1.ipynb)"
        ) from error
    clase = getattr(rfdetr, parametros["clase"], None)
    if clase is None:
        raise ValueError(f"rfdetr no tiene el modelo {parametros['clase']!r}")

    dispositivo = parametros["dispositivo"]
    if dispositivo == "auto":
        import torch

        dispositivo = "cuda" if torch.cuda.is_available() else "cpu"

    clase().train(
        dataset_dir=str(dataset_dir),
        output_dir=str(salida),
        epochs=parametros["epocas"],
        batch_size=parametros["batch_size"],
        grad_accum_steps=parametros["grad_accum_steps"],
        lr=parametros["lr"],
        lr_encoder=parametros["lr_encoder"],
        resolution=parametros["resolucion"],
        seed=parametros["semilla"],
        num_workers=parametros["num_workers"],
        amp_dtype=parametros["amp_dtype"],
        early_stopping=parametros["early_stopping"],
        checkpoint_interval=parametros["checkpoint_interval"],
        tensorboard=parametros["tensorboard"],
        class_names=list(parametros["clases"]),
        device=dispositivo,
    )


def _memoria_gpu_max_mb() -> float | None:
    """Pico de memoria de la GPU durante el entrenamiento (MB), o ``None`` sin torch o sin GPU."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return round(torch.cuda.max_memory_allocated() / 1024**2, 1)
    except Exception:  # sin torch, sin GPU o sin CUDA: es solo un dato informativo
        return None


def _escribir_json(ruta: Path, datos) -> None:
    ruta.write_text(json.dumps(datos, ensure_ascii=False, indent=1), encoding="utf-8")


def _registro(env: dict, mejor: dict | None) -> dict:
    """Campos de la cabecera de la nota de experimento (para ``registro campos --set ...``)."""
    uso = "gpu" if env.get("gpu") else "cpu"
    return {
        "commit": env.get("commit"),
        "entorno": f"{env.get('plataforma') or 'local'}-{uso}",
        "python": env.get("python"),
        "torch": env.get("torch"),
        "metrica_principal": METRICA_PRINCIPAL,
        "valor": mejor["AP50_95"] if mejor else None,
    }


def _linea_registro(env: dict) -> str:
    """La línea ``REGISTRO:`` para copiar a las observaciones (como en el notebook de la v0)."""
    return (
        f"REGISTRO: commit={env.get('commit') or '?'} | GPU={env.get('gpu') or 'ninguna'} | "
        f"rfdetr={env.get('rfdetr') or '?'} | Python={env.get('python') or '?'} | "
        f"PyTorch={env.get('torch') or '?'}"
    )


def _comando_registro(registro: dict) -> str:
    """El comando de PowerShell que anota estos datos en la nota del experimento."""
    pares = [("commit", registro["commit"]), ("entorno", registro["entorno"]),
             ("python", registro["python"]), ("torch", registro["torch"])]  # fmt: skip
    if registro["valor"] is not None:
        pares += [("metrica_principal", registro["metrica_principal"]),
                  ("valor", registro["valor"])]  # fmt: skip
    sets = " ".join(f"--set {clave}={valor}" for clave, valor in pares if valor is not None)
    return f"python -m src.experimentos.registro campos --exp expNNN {sets}"


def ejecutar(
    config: ConfigEntrenamiento,
    exp_id: str,
    *,
    smoke: bool = False,
    raiz: Path = RAIZ,
    trabajo: Path | None = None,
    entrenador=None,
    entorno=None,
    reloj=time.perf_counter,
    memoria_gpu=None,
) -> dict:
    """Prepara los datos, entrena y guarda todo lo de la ejecución en ``runs/<exp_id>/``.

    En lenguaje llano, en orden: comprueba el ``exp_id`` y que la ejecución no exista ya; crea la
    copia de trabajo del dataset (con el subconjunto del smoke y las repeticiones de svodk) fuera
    de ``runs/``; guarda la configuración usada, el entorno y el informe del dataset; entrena
    midiendo el tiempo; mueve los mejores pesos a ``pesos/``; lee las métricas de valid del CSV de
    rfdetr y guarda por época y el mejor punto; y escribe ``resumen.json``. Imprime la línea
    ``REGISTRO:`` y el comando para anotarlo en la nota del experimento.

    Si el entrenamiento falla, o no deja ``metrics.csv``, se guarda el error en ``resumen.json``
    (sin rutas absolutas), sin inventar métricas, y se relanza la excepción. ``entrenador``,
    ``entorno``, ``reloj`` y ``memoria_gpu`` se pueden sustituir al probar. Devuelve el resumen.
    """
    entrenador = entrenador or entrenar_rfdetr
    entorno = entorno or (lambda: entorno_entrenamiento(raiz=raiz))
    memoria_gpu = memoria_gpu or _memoria_gpu_max_mb

    if not _ID_EJECUCION.fullmatch(exp_id or ""):
        raise ValueError(
            f"exp-id no válido: {exp_id!r} (solo letras, números, '-', '_' y '.', sin espacios "
            "ni barras, y sin empezar por un punto)"
        )
    carpeta = raiz / config.salida / exp_id
    if carpeta.exists() and any(carpeta.iterdir()):
        raise ValueError(
            f"La ejecución '{exp_id}' ya existe en {config.salida}/: usa otro --exp-id "
            "(no se pisa una ejecución anterior)"
        )
    dataset = raiz / config.dataset
    informe_dataset = dataset / "informe.json"
    if not informe_dataset.is_file():
        raise FileNotFoundError(
            f"No existe {config.dataset}/informe.json: genera el dataset con `python -m "
            "src.data.importar_publicos importar --etapa configs/datasets/etapa1_v1.yaml`"
        )

    base_trabajo = Path(trabajo) if trabajo else raiz / config.trabajo
    carpeta.mkdir(parents=True, exist_ok=True)
    recuentos = crear_copia_de_trabajo(
        dataset,
        base_trabajo / exp_id,
        repeticiones_svodk=config.repeticiones_svodk,
        prefijo_svodk=config.prefijo_svodk,
        smoke_imagenes=config.smoke_imagenes_train if smoke else None,
        semilla=config.semilla,
    )

    parametros = _parametros(config, smoke)
    env = entorno()
    usada = config.a_diccionario()
    usada["entrenamiento"]["epocas"] = parametros["epocas"]
    usada["datos"]["trabajo"] = mostrar(base_trabajo, raiz)  # nunca una ruta absoluta
    usada["ejecucion"] = {"exp_id": exp_id, "smoke": smoke}
    (carpeta / "config_usada.yaml").write_text(
        yaml.safe_dump(usada, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    _escribir_json(carpeta / "entorno.json", env)
    shutil.copyfile(informe_dataset, carpeta / "dataset_informe.json")

    resumen = {
        "exp_id": exp_id,
        "smoke": smoke,
        "error": None,
        "tiempo_entrenamiento_s": None,
        "epocas_pedidas": parametros["epocas"],
        "epocas_con_metricas": None,
        "mejor": None,
        "memoria_gpu_max_mb": None,
        "pesos": [],
        "recuentos": recuentos,
        "registro": _registro(env, None),
        "linea_registro": _linea_registro(env),
    }

    def guardar_error(error: Exception) -> None:
        resumen["error"] = sanear(f"{type(error).__name__}: {error}", raiz)
        _escribir_json(carpeta / "resumen.json", resumen)

    salida_rfdetr = carpeta / "rfdetr"
    inicio = reloj()
    try:
        entrenador(parametros, base_trabajo / exp_id, salida_rfdetr)
    except Exception as error:
        resumen["tiempo_entrenamiento_s"] = round(reloj() - inicio, 2)
        guardar_error(error)
        raise
    resumen["tiempo_entrenamiento_s"] = round(reloj() - inicio, 2)
    resumen["memoria_gpu_max_mb"] = memoria_gpu()

    for nombre in PESOS_A_GUARDAR:
        origen = salida_rfdetr / nombre
        if origen.is_file():
            (carpeta / "pesos").mkdir(exist_ok=True)
            destino = shutil.move(str(origen), str(carpeta / "pesos" / nombre))
            resumen["pesos"].append(
                {
                    "archivo": f"pesos/{nombre}",
                    "tamano_mb": round(Path(destino).stat().st_size / 1024**2, 2),
                }
            )

    try:
        metricas = leer_metricas_csv(salida_rfdetr / "metrics.csv")
    except (FileNotFoundError, ValueError) as error:
        guardar_error(error)
        raise
    mejor = mejor_punto(metricas)
    _escribir_json(carpeta / "metricas_valid.json", {"metricas": metricas, "mejor": mejor})
    resumen["epocas_con_metricas"] = len(metricas)
    resumen["mejor"] = mejor
    resumen["registro"] = _registro(env, mejor)
    _escribir_json(carpeta / "resumen.json", resumen)

    print(f"Ejecución guardada en {mostrar(carpeta, raiz)} ({resumen['tiempo_entrenamiento_s']} s)")
    if mejor:
        print(
            f"Mejor punto: época {mejor['epoca']}, AP50:95={mejor['AP50_95']}, AP50={mejor['AP50']}"
        )
    print(resumen["linea_registro"])
    if env.get("dirty"):
        print(
            "AVISO: el código tiene cambios sin commitear (dirty): esta ejecución no se "
            "reproduce solo con el commit."
        )
    if smoke:
        print("Ejecución smoke: es solo una prueba rápida, no se registra como experimento.")
    else:
        print(_comando_registro(resumen["registro"]))
    return resumen


def crear_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.train.entrenar",
        description="Entrena RF-DETR para la etapa 1 y guarda la ejecución en runs/<exp-id>/.",
    )
    parser.add_argument(
        "--config", type=Path, default=CONFIG_POR_DEFECTO, help="YAML de entrenamiento"
    )
    parser.add_argument("--exp-id", required=True, help="nombre de la carpeta runs/<exp-id>/")
    parser.add_argument(
        "--smoke", action="store_true", help="1 época, unas 20 imágenes de train y todo valid"
    )
    parser.add_argument(
        "--raiz", type=Path, default=RAIZ, help="raíz a la que son relativas las rutas"
    )
    parser.add_argument(
        "--trabajo", type=Path, default=None,
        help="carpeta de las copias de trabajo del dataset (por defecto, datos.trabajo del YAML)",
    )  # fmt: skip
    return parser


def main(argv: list[str] | None = None, **inyecciones) -> int:
    """Ejecuta el entrenamiento. Devuelve 0 si va bien y 1 si hay un error (con mensaje).

    ``inyecciones`` (``entrenador``, ``entorno``, ``reloj``, ``memoria_gpu``) se pasan a
    ``ejecutar``: sirven para probarlo sin entrenar de verdad.
    """
    for flujo in (sys.stdout, sys.stderr):
        if hasattr(flujo, "reconfigure"):
            flujo.reconfigure(errors="replace")
    args = crear_parser().parse_args(argv)
    try:
        config = cargar_config_entrenamiento(args.config)
        ejecutar(
            config, args.exp_id, smoke=args.smoke, raiz=args.raiz, trabajo=args.trabajo,
            **inyecciones,
        )  # fmt: skip
    except (ValueError, OSError, ImportError, RuntimeError) as error:
        print(f"ERROR: {sanear(str(error), args.raiz)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
