"""Configuración del entrenamiento (``configs/train/*.yaml``): lectura y validación.

Todo parámetro del entrenamiento sale de aquí: el código no lleva valores fijos. Un YAML mal
escrito (falta una clave, un valor fuera de rango o una clave con una errata) da un error que
nombra la clave, en lugar de entrenar con algo distinto de lo que se quería.
"""

from dataclasses import dataclass
from pathlib import Path

from src.config import load_config

# Modelos de rfdetr con licencia Apache-2.0. RFDETRXLarge y RFDETR2XLarge viven en el paquete
# aparte rfdetr_plus (licencia PML) y no se pueden usar en este proyecto.
MODELOS_APACHE = ("RFDETRNano", "RFDETRSmall", "RFDETRMedium")
MODELOS_PLUS = ("RFDETRXLarge", "RFDETR2XLarge")

AMP_DTYPES = ("auto", "fp16", "bf16")
DISPOSITIVOS = ("auto", "cpu", "cuda")

# Claves esperadas de cada sección. Cualquier otra es un error (casi seguro una errata).
_ESQUEMA = {
    "modelo": ("clase", "resolucion", "clases"),
    "entrenamiento": (
        "epocas", "batch_size", "grad_accum_steps", "lr", "lr_encoder", "amp_dtype",
        "num_workers", "early_stopping", "checkpoint_interval", "tensorboard", "dispositivo",
    ),
    "datos": ("dataset", "repeticiones_svodk", "prefijo_svodk", "trabajo"),
    "salida": ("carpeta",),
    "smoke": ("epocas", "imagenes_train"),
}  # fmt: skip
_CLAVES_RAIZ = ("semilla", *_ESQUEMA)


@dataclass(frozen=True)
class ConfigEntrenamiento:
    """Parámetros de un entrenamiento, ya validados."""

    modelo_clase: str
    resolucion: int
    clases: tuple[str, ...]
    epocas: int
    batch_size: int
    grad_accum_steps: int
    lr: float
    lr_encoder: float
    amp_dtype: str
    num_workers: int
    early_stopping: bool
    checkpoint_interval: int
    tensorboard: bool
    dispositivo: str
    semilla: int
    dataset: str  # carpeta del dataset exportado, relativa a la raíz del repositorio
    repeticiones_svodk: int
    prefijo_svodk: str
    trabajo: str  # carpeta donde se crean las copias de trabajo del dataset
    salida: str  # carpeta de las ejecuciones (runs/<exp_id>/)
    smoke_epocas: int
    smoke_imagenes_train: int

    def a_diccionario(self) -> dict:
        """La configuración con la misma forma que el YAML (para guardarla con la ejecución)."""
        return {
            "modelo": {
                "clase": self.modelo_clase,
                "resolucion": self.resolucion,
                "clases": list(self.clases),
            },
            "entrenamiento": {
                "epocas": self.epocas,
                "batch_size": self.batch_size,
                "grad_accum_steps": self.grad_accum_steps,
                "lr": self.lr,
                "lr_encoder": self.lr_encoder,
                "amp_dtype": self.amp_dtype,
                "num_workers": self.num_workers,
                "early_stopping": self.early_stopping,
                "checkpoint_interval": self.checkpoint_interval,
                "tensorboard": self.tensorboard,
                "dispositivo": self.dispositivo,
            },
            "semilla": self.semilla,
            "datos": {
                "dataset": self.dataset,
                "repeticiones_svodk": self.repeticiones_svodk,
                "prefijo_svodk": self.prefijo_svodk,
                "trabajo": self.trabajo,
            },
            "salida": {"carpeta": self.salida},
            "smoke": {"epocas": self.smoke_epocas, "imagenes_train": self.smoke_imagenes_train},
        }


def _entero(valor, clave: str, minimo: int):
    """Un entero (no booleano) mayor o igual que ``minimo``; si no, ``ValueError``."""
    if isinstance(valor, bool) or not isinstance(valor, int) or valor < minimo:
        raise ValueError(f"{clave}: debe ser un entero >= {minimo} (llegó {valor!r})")
    return valor


def _positivo(valor, clave: str):
    """Un número (no booleano) mayor que cero; si no, ``ValueError``."""
    if isinstance(valor, bool) or not isinstance(valor, int | float) or valor <= 0:
        raise ValueError(f"{clave}: debe ser un número > 0 (llegó {valor!r})")
    return valor


def _booleano(valor, clave: str) -> bool:
    if not isinstance(valor, bool):
        raise ValueError(f"{clave}: debe ser true o false (llegó {valor!r})")
    return valor


def _texto(valor, clave: str) -> str:
    if not isinstance(valor, str) or not valor.strip():
        raise ValueError(f"{clave}: debe ser un texto no vacío (llegó {valor!r})")
    return valor


def _opcion(valor, clave: str, validas: tuple[str, ...]) -> str:
    if valor not in validas:
        raise ValueError(f"{clave}: debe ser uno de {', '.join(validas)} (llegó {valor!r})")
    return valor


def _modelo(valor, clave: str) -> str:
    """Solo modelos Apache-2.0; los de rfdetr_plus se rechazan con la razón."""
    if valor in MODELOS_PLUS:
        raise ValueError(
            f"{clave}: {valor} pertenece a rfdetr_plus (licencia PML) y no se puede usar; "
            f"modelos permitidos: {', '.join(MODELOS_APACHE)}"
        )
    if valor not in MODELOS_APACHE:
        raise ValueError(
            f"{clave}: modelo {valor!r} no admitido; válidos (Apache-2.0): "
            f"{', '.join(MODELOS_APACHE)}"
        )
    return valor


def _clases(valor, clave: str) -> tuple[str, ...]:
    if (
        not isinstance(valor, list)
        or not valor
        or not all(isinstance(c, str) and c.strip() for c in valor)
    ):
        raise ValueError(f"{clave}: debe ser una lista no vacía de nombres (llegó {valor!r})")
    return tuple(valor)


def _comprobar_claves(config: dict) -> None:
    """Falta una sección o clave, o hay una desconocida -> ``ValueError`` que la nombra."""
    for clave in _CLAVES_RAIZ:
        if clave not in config:
            raise ValueError(f"falta '{clave}' en la configuración")
    desconocidas = [c for c in config if c not in _CLAVES_RAIZ]
    for seccion, claves in _ESQUEMA.items():
        datos = config[seccion]
        if not isinstance(datos, dict):
            raise ValueError(f"'{seccion}' debe ser un bloque con claves (llegó {datos!r})")
        for clave in claves:
            if clave not in datos:
                raise ValueError(f"falta '{seccion}.{clave}' en la configuración")
        desconocidas += [f"{seccion}.{c}" for c in datos if c not in claves]
    if desconocidas:
        raise ValueError(
            f"claves desconocidas en la configuración: {', '.join(map(str, desconocidas))}"
        )


def cargar_config_entrenamiento(ruta: str | Path) -> ConfigEntrenamiento:
    """Lee y valida un YAML de entrenamiento.

    En lenguaje llano: abre el YAML, comprueba que están todas las secciones y claves (y ninguna
    de más), que cada valor tiene el tipo y el rango correctos y que el modelo es de licencia
    Apache-2.0, y lo devuelve como un objeto con los valores ya convertidos. Si algo falla, el
    error dice qué clave es (por ejemplo ``entrenamiento.epocas``).

    Errores: ``FileNotFoundError`` si el archivo no existe; ``ValueError`` si el contenido no es
    válido.
    """
    config = load_config(ruta)
    _comprobar_claves(config)
    modelo, entrenamiento = config["modelo"], config["entrenamiento"]
    datos, salida, smoke = config["datos"], config["salida"], config["smoke"]
    return ConfigEntrenamiento(
        modelo_clase=_modelo(modelo["clase"], "modelo.clase"),
        resolucion=_entero(modelo["resolucion"], "modelo.resolucion", 1),
        clases=_clases(modelo["clases"], "modelo.clases"),
        epocas=_entero(entrenamiento["epocas"], "entrenamiento.epocas", 1),
        batch_size=_entero(entrenamiento["batch_size"], "entrenamiento.batch_size", 1),
        grad_accum_steps=_entero(
            entrenamiento["grad_accum_steps"], "entrenamiento.grad_accum_steps", 1
        ),
        lr=_positivo(entrenamiento["lr"], "entrenamiento.lr"),
        lr_encoder=_positivo(entrenamiento["lr_encoder"], "entrenamiento.lr_encoder"),
        amp_dtype=_opcion(entrenamiento["amp_dtype"], "entrenamiento.amp_dtype", AMP_DTYPES),
        num_workers=_entero(entrenamiento["num_workers"], "entrenamiento.num_workers", 0),
        early_stopping=_booleano(entrenamiento["early_stopping"], "entrenamiento.early_stopping"),
        checkpoint_interval=_entero(
            entrenamiento["checkpoint_interval"], "entrenamiento.checkpoint_interval", 1
        ),
        tensorboard=_booleano(entrenamiento["tensorboard"], "entrenamiento.tensorboard"),
        dispositivo=_opcion(
            entrenamiento["dispositivo"], "entrenamiento.dispositivo", DISPOSITIVOS
        ),
        semilla=_entero(config["semilla"], "semilla", 0),
        dataset=_texto(datos["dataset"], "datos.dataset"),
        repeticiones_svodk=_entero(datos["repeticiones_svodk"], "datos.repeticiones_svodk", 1),
        prefijo_svodk=_texto(datos["prefijo_svodk"], "datos.prefijo_svodk"),
        trabajo=_texto(datos["trabajo"], "datos.trabajo"),
        salida=_texto(salida["carpeta"], "salida.carpeta"),
        smoke_epocas=_entero(smoke["epocas"], "smoke.epocas", 1),
        smoke_imagenes_train=_entero(smoke["imagenes_train"], "smoke.imagenes_train", 1),
    )
