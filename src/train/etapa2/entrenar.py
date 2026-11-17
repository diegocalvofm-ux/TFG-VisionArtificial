"""Entrenamiento del clasificador de la etapa 2 y registro de la ejecución en ``runs/<exp_id>/``.

Uso (desde la raíz del repositorio):
    python -m src.train.etapa2 --config configs/train/etapa2_effb0.yaml --exp-id mi_prueba --smoke
    python -m src.train.etapa2 --config configs/train/etapa2_effb0.yaml --exp-id etapa2_effb0_001

``--smoke`` entrena ``smoke.epocas`` (1) época con unos pocos recortes de train, valid y test (una
prueba rápida de que todo funciona). Cada ejecución crea ``runs/<exp_id>/`` con:

- ``config_usada.yaml``, ``entorno.json`` (versiones, GPU, commit) y ``dataset_informe.json``;
- ``datos.json``: recortes por partición, fuente y etiqueta, sobremuestreo, pesos de clase...;
- ``metricas_por_epoca.json``: pérdidas y métricas de valid de cada época (se escribe cada época);
- ``resumen.json``: mejor época, métricas de valid y test con el mejor punto frente al
  clasificador mayoritario, tiempo, memoria de la GPU, avisos y los campos para el registro;
- ``matrices_confusion.json`` y ``pesos/mejor.pt`` y ``pesos/ultimo.pt``.

El mejor punto es la época con mayor media de la exactitud equilibrada en valid de los grupos que
tienen datos en valid. El dataset solo se lee. Todo parámetro sale del YAML (``config.py``).
"""

import argparse
import contextlib
import math
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from src.data.schema import GROUPS
from src.experimentos.privacidad import mostrar, sanear
from src.train.entorno import entorno_entrenamiento, versiones_paquetes
from src.train.entrenar import _ID_EJECUCION, _escribir_json, _memoria_gpu_max_mb
from src.train.etapa2.aumentos import DatasetRecortes, transformacion_eval, transformacion_train
from src.train.etapa2.config import ConfigEtapa2, cargar_config_etapa2
from src.train.etapa2.datos import (
    PARTICIONES,
    codificar,
    indices_por_epoca,
    leer_recortes,
    recuentos,
    repartir_propio,
    subconjunto,
)
from src.train.etapa2.metricas import clases_mayoritarias, evaluar_conjunto, separar_matrices
from src.train.etapa2.modelo import crear_extractor_timm, crear_modelo, guardar_pesos
from src.train.etapa2.perdida import perdida_enmascarada, pesos_por_clase
from src.utils.env import _sin_datos_personales

RAIZ = Path(__file__).resolve().parents[3]
CONFIG_POR_DEFECTO = RAIZ / "configs" / "train" / "etapa2_effb0.yaml"

# Nombre con el que se registra la métrica del experimento: el criterio del mejor punto.
METRICA_PRINCIPAL = "exactitud_equilibrada_media_valid"
PESOS = ("mejor.pt", "ultimo.pt")


# --- Piezas sueltas (se prueban por separado) ---------------------------------------------


def factor_lr(paso: int, total: int, calentamiento: int, factor_final: float) -> float:
    """Por cuánto se multiplica la tasa de aprendizaje en el paso ``paso`` (empieza en 0).

    En lenguaje llano: durante los ``calentamiento`` primeros pasos la tasa sube en línea recta
    hasta su valor (empezar con la tasa entera puede estropear los pesos preentrenados); después
    baja siguiendo medio coseno hasta ``factor_final`` en el último paso (``total - 1``), primero
    despacio, luego deprisa y al final otra vez despacio.
    """
    if paso < calentamiento:
        return (paso + 1) / calentamiento
    tramo = total - calentamiento - 1
    progreso = 0.0 if tramo <= 0 else min(1.0, (paso - calentamiento) / tramo)
    return factor_final + (1 - factor_final) * 0.5 * (1 + math.cos(math.pi * progreso))


def pasos_calentamiento(epocas_calentamiento: int, pasos_por_epoca: int, total: int) -> int:
    """Pasos de calentamiento: las épocas del YAML, pero nunca más de la mitad del total.

    El límite hace que un smoke de una sola época no se pase entero calentando.
    """
    return min(epocas_calentamiento * pasos_por_epoca, total // 2)


def resolver_dispositivo(nombre: str) -> str:
    """``auto`` -> ``cuda`` si hay GPU y si no ``cpu``. Pedir ``cuda`` sin GPU es un error."""
    hay_gpu = torch.cuda.is_available()
    if nombre == "auto":
        return "cuda" if hay_gpu else "cpu"
    if nombre == "cuda" and not hay_gpu:
        raise RuntimeError("entrenamiento.dispositivo es cuda pero no hay GPU disponible")
    return nombre


def fijar_semilla(semilla: int) -> None:
    """Misma semilla en random, numpy y torch: misma inicialización, orden y aumentos.

    En CPU dos ejecuciones con la misma semilla dan lo mismo; en GPU algunas operaciones de cuDNN
    no son deterministas y el resultado puede variar un poco.
    """
    random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)


def _semilla_trabajador(_indice: int) -> None:
    """Semilla de cada proceso del ``DataLoader`` (torch ya da una distinta a cada uno)."""
    semilla = torch.initial_seed() % 2**32
    np.random.seed(semilla)
    random.seed(semilla)


def entorno_etapa2(raiz: Path = RAIZ) -> dict:
    """El entorno de la etapa 1 (versiones, GPU, commit...) más la versión de timm."""
    env = entorno_entrenamiento(raiz=raiz)
    env["timm"] = _sin_datos_personales(versiones_paquetes(("timm",))["timm"])
    return env


# --- Datos ---------------------------------------------------------------------------------


def preparar_datos(config: ConfigEtapa2, carpeta: Path, smoke: bool) -> dict:
    """Lee los recortes y decide qué filas van a cada partición y cuántas veces se usan.

    En lenguaje llano, en este orden: lee y valida ``recortes.csv``; si el YAML lo pide, reparte
    de nuevo la fuente propia por lechuga; en el smoke se queda con unos pocos recortes de cada
    partición; codifica las etiquetas; aplica el sobremuestreo (solo a train); calcula los pesos de
    clase con train tal como lo ve la pérdida en una época (con las repeticiones, para no
    compensar dos veces la fuente propia) y las clases mayoritarias con train sin repeticiones.

    Devuelve ``{"tablas", "objetivos", "indices_train", "mayoritarias", "pesos", "resumen"}``;
    ``resumen`` es lo que se guarda en ``datos.json``. Error (``ValueError``) si train o valid se
    quedan vacíos.
    """
    tabla, sin_etiquetas = leer_recortes(carpeta)
    reparto = None
    if config.reparto_propio is not None:
        tabla, reparto = repartir_propio(tabla, config.fuente_propia, config.reparto_propio,
                                         config.semilla)  # fmt: skip
    tablas = {p: tabla[tabla["particion"] == p].reset_index(drop=True) for p in PARTICIONES}
    if smoke:
        tablas["train"] = subconjunto(tablas["train"], config.smoke_recortes_train, config.semilla)
        for particion in ("valid", "test"):
            tablas[particion] = subconjunto(tablas[particion], config.smoke_recortes_eval,
                                            config.semilla)  # fmt: skip
    for particion in ("train", "valid"):
        if tablas[particion].empty:
            raise ValueError(f"no hay recortes de {particion} en {config.dataset}: hacen falta "
                             "para entrenar y para elegir el mejor punto")  # fmt: skip

    objetivos = {p: codificar(t) for p, t in tablas.items()}
    indices = indices_por_epoca(tablas["train"], config.fuente_propia, config.sobremuestreo_propio)
    vistos = objetivos["train"][indices]
    pesos, sin_muestras = {}, {}
    for j, (grupo, etiquetas) in enumerate(GROUPS.items()):
        columna = vistos[:, j]
        cuentas = np.bincount(columna[columna >= 0], minlength=len(etiquetas)).tolist()
        pesos[grupo] = dict(zip(etiquetas, pesos_por_clase(cuentas, config.pesos_clase),
                                strict=True))  # fmt: skip
        sin_muestras[grupo] = [e for e, n in zip(etiquetas, cuentas, strict=True) if n == 0]
    mayoritarias = clases_mayoritarias(objetivos["train"])

    avisos = []
    for particion in ("valid", "test"):
        sin_datos = [g for j, g in enumerate(GROUPS) if not (objetivos[particion][:, j] >= 0).any()]
        if tablas[particion].empty:
            avisos.append(f"{particion} está vacío: no hay métricas de {particion}")
        elif sin_datos:
            con_datos = [g for g in GROUPS if g not in sin_datos]
            avisos.append(
                f"{particion} no tiene muestras visibles de {', '.join(sin_datos)}: esos grupos no "
                f"tienen métricas de {particion}"
                + (f" y el criterio del mejor punto solo usa {', '.join(con_datos)}"
                   if particion == "valid" else "")  # fmt: skip
            )
    ausentes = [f"{g} ({', '.join(e)})" for g, e in sin_muestras.items() if e]
    if ausentes:
        avisos.append("clases sin muestras en train (peso 0: el modelo no puede aprenderlas): "
                      + "; ".join(ausentes))  # fmt: skip
    hay_propias = (tablas["train"]["fuente"] == config.fuente_propia).any()
    if config.sobremuestreo_propio > 1 and not hay_propias:
        avisos.append(f"sobremuestreo_propio = {config.sobremuestreo_propio} pero no hay recortes "
                      f"de '{config.fuente_propia}' en train")  # fmt: skip
    if sin_etiquetas:
        avisos.append(f"{len(sin_etiquetas)} recortes sin ningún grupo visible: no se usan")

    resumen = {
        "dataset": config.dataset,
        "smoke": smoke,
        "filas_sin_etiquetas": sin_etiquetas,
        "reparto_propio": reparto,
        "recuentos": recuentos(pd.concat(tablas.values(), ignore_index=True)),
        "sobremuestreo_propio": config.sobremuestreo_propio,
        "muestras_por_epoca": len(indices),
        "pesos_clase": pesos,
        "clases_sin_muestras_train": sin_muestras,
        "clases_mayoritarias": mayoritarias,
        "avisos": avisos,
    }
    pesos_tensor = {g: torch.tensor(list(w.values()), dtype=torch.float32)
                    for g, w in pesos.items()}  # fmt: skip
    return {"tablas": tablas, "objetivos": objetivos, "indices_train": indices,
            "mayoritarias": mayoritarias, "pesos": pesos_tensor, "resumen": resumen}  # fmt: skip


def _cargador(rutas, objetivos, transformacion, config: ConfigEtapa2, dispositivo: str, *,
              barajar: bool) -> DataLoader:  # fmt: skip
    generador = torch.Generator().manual_seed(config.semilla) if barajar else None
    return DataLoader(
        DatasetRecortes(rutas, objetivos, transformacion),
        batch_size=config.batch_size,
        shuffle=barajar,
        num_workers=config.num_workers,
        pin_memory=dispositivo == "cuda",
        persistent_workers=config.num_workers > 0,
        worker_init_fn=_semilla_trabajador,
        generator=generador,
    )


# --- Bucle ---------------------------------------------------------------------------------


def _autocast(dispositivo: str, usar_amp: bool):
    """fp16 automático en GPU; en CPU (o sin precisión mixta), nada."""
    if usar_amp:
        return torch.autocast(device_type=dispositivo, dtype=torch.float16)
    return contextlib.nullcontext()


def entrenar_epoca(modelo, cargador, optimizador, programador, escalador, pesos, dispositivo,
                   usar_amp) -> dict:  # fmt: skip
    """Una pasada por train. Devuelve la pérdida media, los pasos y los lotes sin etiquetas.

    En lenguaje llano: para cada lote, el modelo predice, se calcula la pérdida enmascarada (con
    los pesos de clase), se propagan los gradientes y se actualizan los pesos; la tasa avanza un
    paso aunque el lote no tuviera ninguna etiqueta visible (ese lote no actualiza nada). En GPU
    la pasada hacia delante va en fp16 y ``escalador`` evita que los gradientes pequeños se pierdan.
    """
    modelo.train()
    suma, usados, sin_etiquetas, pasos = 0.0, 0, 0, 0
    for imagenes, objetivos, _ in cargador:
        pasos += 1
        imagenes = imagenes.to(dispositivo, non_blocking=True)
        objetivos = objetivos.to(dispositivo, non_blocking=True)
        with _autocast(dispositivo, usar_amp):
            salidas = modelo(imagenes)
        salidas = {g: s.float() for g, s in salidas.items()}  # la pérdida siempre en fp32
        total, _ = perdida_enmascarada(salidas, objetivos, pesos)
        optimizador.zero_grad(set_to_none=True)
        if total is None:
            sin_etiquetas += 1
        elif escalador is not None:
            escalador.scale(total).backward()
            escalador.step(optimizador)
            escalador.update()
        else:
            total.backward()
            optimizador.step()
        if total is not None:
            suma += float(total.detach())
            usados += 1
        programador.step()
    return {"perdida_train": suma / usados if usados else None, "pasos": pasos,
            "pasos_sin_etiquetas": sin_etiquetas}  # fmt: skip


@torch.no_grad()
def predecir_conjunto(modelo, cargador, n: int, dispositivo: str, usar_amp: bool):
    """Predicción de cada recorte de un conjunto y su pérdida media sin pesos (informativa).

    Devuelve ``(predicciones, perdida)``: matriz (n, 6) con el índice de la etiqueta más probable
    de cada grupo y la media por lote de la pérdida enmascarada (``None`` si no hubo visibles).
    """
    modelo.eval()
    predicciones = np.full((n, len(GROUPS)), -1, dtype=np.int64)
    suma, usados = 0.0, 0
    for imagenes, objetivos, indices in cargador:
        with _autocast(dispositivo, usar_amp):
            salidas = modelo(imagenes.to(dispositivo, non_blocking=True))
        salidas = {g: s.float() for g, s in salidas.items()}
        total, _ = perdida_enmascarada(salidas, objetivos.to(dispositivo))
        if total is not None:
            suma += float(total)
            usados += 1
        filas = indices.numpy()
        for j, logits in enumerate(salidas.values()):
            predicciones[filas, j] = logits.argmax(dim=1).cpu().numpy()
    return predicciones, (suma / usados if usados else None)


def _compacta(evaluacion: dict) -> dict:
    """Lo de valid que se guarda por época: criterio y, por grupo, n y las dos métricas."""
    return {
        "criterio": evaluacion["criterio"],
        "grupos": {g: {k: d["modelo"][k] for k in ("n", "exactitud_equilibrada", "f1_macro")}
                   for g, d in evaluacion["grupos"].items()},  # fmt: skip
    }


def _texto(valor) -> str:
    return "n/a" if valor is None else f"{valor:.4f}"


# --- Registro ------------------------------------------------------------------------------


def _registro(env: dict, mejor: dict | None) -> dict:
    """Campos de la cabecera de la nota de experimento (para ``registro campos --set ...``)."""
    uso = "gpu" if env.get("gpu") else "cpu"
    return {
        "commit": env.get("commit"),
        "entorno": f"{env.get('plataforma') or 'local'}-{uso}",
        "python": env.get("python"),
        "torch": env.get("torch"),
        "metrica_principal": METRICA_PRINCIPAL,
        "valor": mejor["criterio"] if mejor else None,
    }


def _linea_registro(env: dict) -> str:
    """La línea ``REGISTRO:`` para copiar a las observaciones de la nota."""
    return (
        f"REGISTRO: commit={env.get('commit') or '?'} | GPU={env.get('gpu') or 'ninguna'} | "
        f"timm={env.get('timm') or '?'} | Python={env.get('python') or '?'} | "
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


def _pesos_guardados(carpeta: Path) -> list[dict]:
    return [{"archivo": f"pesos/{nombre}",
             "tamano_mb": round((carpeta / "pesos" / nombre).stat().st_size / 1024**2, 2)}
            for nombre in PESOS if (carpeta / "pesos" / nombre).is_file()]  # fmt: skip


# --- Orquestación --------------------------------------------------------------------------


def ejecutar(
    config: ConfigEtapa2,
    exp_id: str,
    *,
    smoke: bool = False,
    raiz: Path = RAIZ,
    crear_extractor=None,
    entorno=None,
    reloj=time.perf_counter,
    memoria_gpu=None,
) -> dict:
    """Prepara los datos, entrena, evalúa y guarda todo lo de la ejecución en ``runs/<exp_id>/``.

    En lenguaje llano, en orden:

    1. Comprueba el ``exp_id`` y que la ejecución no exista ya, y prepara los datos (si algo falla
       aquí no se crea la carpeta).
    2. Guarda la configuración usada, el entorno, el informe del dataset y ``datos.json``.
    3. Crea el modelo (timm descarga los pesos de ImageNet) y entrena época a época: tras cada una
       evalúa valid, guarda ``ultimo.pt`` y, si el criterio mejora, ``mejor.pt`` (en un empate se
       queda la primera), y reescribe ``metricas_por_epoca.json``.
    4. Con los pesos de ``mejor.pt`` evalúa valid y test (modelo frente a clasificador mayoritario,
       global y por fuente) y escribe ``matrices_confusion.json`` y ``resumen.json``.
    5. Imprime la línea ``REGISTRO:`` y el comando para anotar el experimento (en el smoke no).

    Si algo falla desde el paso 3, el error (sin rutas absolutas) queda en ``resumen.json``, sin
    inventar métricas, y se relanza la excepción. ``crear_extractor``, ``entorno``, ``reloj`` y
    ``memoria_gpu`` se pueden sustituir al probar. Devuelve el resumen.
    """
    crear_extractor = crear_extractor or crear_extractor_timm
    entorno = entorno or (lambda: entorno_etapa2(raiz))
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
    datos = preparar_datos(config, dataset, smoke)
    dispositivo = resolver_dispositivo(config.dispositivo)
    usar_amp = config.precision_mixta and dispositivo == "cuda"
    epocas = config.smoke_epocas if smoke else config.epocas
    tablas = datos["tablas"]

    carpeta.mkdir(parents=True, exist_ok=True)
    env = entorno()
    usada = config.a_diccionario()
    usada["entrenamiento"]["epocas"] = epocas
    usada["ejecucion"] = {"exp_id": exp_id, "smoke": smoke}
    (carpeta / "config_usada.yaml").write_text(
        yaml.safe_dump(usada, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    _escribir_json(carpeta / "entorno.json", env)
    if (dataset / "informe.json").is_file():
        shutil.copyfile(dataset / "informe.json", carpeta / "dataset_informe.json")
    _escribir_json(carpeta / "datos.json", datos["resumen"])

    resumen = {
        "exp_id": exp_id,
        "smoke": smoke,
        "error": None,
        "dispositivo": dispositivo,
        "precision_mixta": usar_amp,
        "tiempo_entrenamiento_s": None,
        "epocas_pedidas": epocas,
        "epocas_completadas": 0,
        "mejor": None,
        "valid": None,
        "test": None,
        "memoria_gpu_max_mb": None,
        "pesos": [],
        "muestras": {"train_por_epoca": len(datos["indices_train"]),
                     "valid": len(tablas["valid"]), "test": len(tablas["test"])},  # fmt: skip
        "avisos": list(datos["resumen"]["avisos"]),
        "registro": _registro(env, None),
        "linea_registro": _linea_registro(env),
    }
    inicio = reloj()
    try:
        _entrenar_y_evaluar(config, datos, carpeta, resumen, epocas=epocas,
                            dispositivo=dispositivo, usar_amp=usar_amp,
                            crear_extractor=crear_extractor, reloj=reloj)  # fmt: skip
    except Exception as error:
        resumen["tiempo_entrenamiento_s"] = round(reloj() - inicio, 2)
        resumen["pesos"] = _pesos_guardados(carpeta)
        resumen["error"] = sanear(f"{type(error).__name__}: {error}", raiz)
        _escribir_json(carpeta / "resumen.json", resumen)
        raise
    resumen["tiempo_entrenamiento_s"] = round(reloj() - inicio, 2)
    resumen["memoria_gpu_max_mb"] = memoria_gpu()
    resumen["pesos"] = _pesos_guardados(carpeta)
    resumen["registro"] = _registro(env, resumen["mejor"])
    _escribir_json(carpeta / "resumen.json", resumen)
    _imprimir_final(resumen, carpeta, raiz, env, smoke)
    return resumen


def _entrenar_y_evaluar(config, datos, carpeta, resumen, *, epocas, dispositivo, usar_amp,
                        crear_extractor, reloj) -> None:  # fmt: skip
    """Pasos 3 y 4 de ``ejecutar``: rellena ``resumen`` y escribe pesos, métricas y matrices."""
    tablas, objetivos, indices = datos["tablas"], datos["objetivos"], datos["indices_train"]
    fijar_semilla(config.semilla)
    if dispositivo == "cuda":
        torch.backends.cudnn.benchmark = True  # entrada de tamaño fijo: cuDNN elige lo más rápido
        torch.cuda.reset_peak_memory_stats()
    modelo = crear_modelo(config.arquitectura, config.preentrenado, config.dropout,
                          config.resolucion, crear_extractor).to(dispositivo)  # fmt: skip

    train = tablas["train"]
    cargador_train = _cargador([train["ruta"][i] for i in indices], objetivos["train"][indices],
                               transformacion_train(config.resolucion, config.aumentos), config,
                               dispositivo, barajar=True)  # fmt: skip
    eval_ = transformacion_eval(config.resolucion)
    cargadores = {p: _cargador(list(tablas[p]["ruta"]), objetivos[p], eval_, config, dispositivo,
                               barajar=False)  # fmt: skip
                  for p in ("valid", "test") if not tablas[p].empty}

    optimizador = torch.optim.AdamW(
        [{"params": modelo.extractor.parameters(), "lr": config.lr_extractor},
         {"params": modelo.cabezas.parameters(), "lr": config.lr_cabezas}],
        weight_decay=config.decaimiento_pesos,
    )  # fmt: skip
    pasos_epoca = len(cargador_train)
    total = pasos_epoca * epocas
    calentamiento = pasos_calentamiento(config.epocas_calentamiento, pasos_epoca, total)
    programador = torch.optim.lr_scheduler.LambdaLR(
        optimizador, lambda paso: factor_lr(paso, total, calentamiento, config.factor_lr_final)
    )
    escalador = torch.amp.GradScaler("cuda") if usar_amp else None
    pesos = {g: w.to(dispositivo) for g, w in datos["pesos"].items()}
    fuentes = {p: list(tablas[p]["fuente"]) for p in PARTICIONES}

    def evaluar(particion: str):
        predicciones, perdida = predecir_conjunto(
            modelo, cargadores[particion], len(tablas[particion]), dispositivo, usar_amp
        )
        return evaluar_conjunto(objetivos[particion], predicciones, fuentes[particion],
                                datos["mayoritarias"]), perdida  # fmt: skip

    historial = []
    mejor_criterio = None
    for epoca in range(1, epocas + 1):
        inicio = reloj()
        entrenamiento = entrenar_epoca(modelo, cargador_train, optimizador, programador,
                                       escalador, pesos, dispositivo, usar_amp)  # fmt: skip
        valid, perdida_valid = evaluar("valid")
        criterio = valid["criterio"]
        metadatos = {"epoca": epoca, "criterio": criterio}
        guardar_pesos(carpeta / "pesos" / "ultimo.pt", modelo, metadatos)
        if criterio is not None and (mejor_criterio is None or criterio > mejor_criterio):
            mejor_criterio = criterio
            guardar_pesos(carpeta / "pesos" / "mejor.pt", modelo, metadatos)
            resumen["mejor"] = {"epoca": epoca, "criterio": criterio}
        historial.append({
            "epoca": epoca,
            **entrenamiento,
            "lr": [grupo["lr"] for grupo in optimizador.param_groups],
            "tiempo_s": round(reloj() - inicio, 2),
            "valid": {"perdida": perdida_valid, **_compacta(valid)},
        })  # fmt: skip
        resumen["epocas_completadas"] = epoca
        _escribir_json(carpeta / "metricas_por_epoca.json",
                       {"criterio": METRICA_PRINCIPAL, "epocas": historial})  # fmt: skip
        print(f"época {epoca}/{epocas}: pérdida train {_texto(entrenamiento['perdida_train'])}, "
              f"valid {_texto(perdida_valid)}, criterio {_texto(criterio)} "
              f"({historial[-1]['tiempo_s']} s)", flush=True)  # fmt: skip

    if resumen["mejor"] is None:  # ninguna época tuvo criterio: no hay datos visibles en valid
        shutil.copyfile(carpeta / "pesos" / "ultimo.pt", carpeta / "pesos" / "mejor.pt")
        resumen["mejor"] = {"epoca": epocas, "criterio": None}
        resumen["avisos"].append("ninguna época tuvo criterio en valid: mejor.pt es la última")

    estado = torch.load(carpeta / "pesos" / "mejor.pt", map_location=dispositivo,
                        weights_only=True)["estado"]  # fmt: skip
    modelo.load_state_dict(estado)
    matrices = {}
    for particion in ("valid", "test"):
        if particion not in cargadores:
            resumen[particion], matrices[particion] = None, None
            continue
        evaluacion, perdida = evaluar(particion)
        resumen[particion], matrices[particion] = separar_matrices(evaluacion)
        resumen[particion]["perdida"] = perdida
    _escribir_json(carpeta / "matrices_confusion.json", matrices)


def _imprimir_final(resumen: dict, carpeta: Path, raiz: Path, env: dict, smoke: bool) -> None:
    print(f"Ejecución guardada en {mostrar(carpeta, raiz)} ({resumen['tiempo_entrenamiento_s']} s)")
    mejor = resumen["mejor"]
    print(f"Mejor punto: época {mejor['epoca']}, exactitud equilibrada media en valid "
          f"{_texto(mejor['criterio'])} (grupos con datos: "
          f"{', '.join(resumen['valid']['grupos_con_datos']) or 'ninguno'})")  # fmt: skip
    if resumen["test"] is not None:
        print(f"Test con el mejor punto: exactitud equilibrada media "
              f"{_texto(resumen['test']['criterio'])}")  # fmt: skip
    for aviso in resumen["avisos"]:
        print(f"AVISO: {aviso}")
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


# --- Línea de comandos ---------------------------------------------------------------------


def crear_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.train.etapa2",
        description="Entrena el clasificador de la etapa 2 y guarda la ejecución en "
        "runs/<exp-id>/.",
    )
    parser.add_argument(
        "--config", type=Path, default=CONFIG_POR_DEFECTO, help="YAML de entrenamiento"
    )
    parser.add_argument("--exp-id", required=True, help="nombre de la carpeta runs/<exp-id>/")
    parser.add_argument(
        "--smoke", action="store_true", help="1 época con unos pocos recortes de cada partición"
    )
    parser.add_argument(
        "--raiz", type=Path, default=RAIZ, help="raíz a la que son relativas las rutas"
    )
    return parser


def main(argv: list[str] | None = None, **inyecciones) -> int:
    """Ejecuta el entrenamiento. Devuelve 0 si va bien y 1 si hay un error (con mensaje).

    ``inyecciones`` (``crear_extractor``, ``entorno``, ``reloj``, ``memoria_gpu``) se pasan a
    ``ejecutar``: sirven para probarlo sin timm ni GPU.
    """
    for flujo in (sys.stdout, sys.stderr):
        if hasattr(flujo, "reconfigure"):
            flujo.reconfigure(errors="replace")
    args = crear_parser().parse_args(argv)
    try:
        config = cargar_config_etapa2(args.config)
        ejecutar(config, args.exp_id, smoke=args.smoke, raiz=args.raiz, **inyecciones)
    except (ValueError, OSError, ImportError, RuntimeError) as error:
        print(f"ERROR: {sanear(str(error), args.raiz)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
