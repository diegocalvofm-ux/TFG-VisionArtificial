"""Configuración del entrenamiento de la etapa 2 (``configs/train/etapa2_*.yaml``).

Igual que en la etapa 1, todo parámetro sale del YAML y un YAML mal escrito (falta una clave,
sobra una con una errata o un valor está fuera de rango) da un error que nombra la clave. Aquí,
además, no hay forma de pedir cambios de tono ni de saturación: falsearían el grupo ``color``.
"""

from dataclasses import dataclass
from pathlib import Path

from src.config import load_config
from src.data.split import _validar_fracciones
from src.eval.config import _numero, _ruta_relativa
from src.train.config import DISPOSITIVOS, _booleano, _entero, _opcion, _positivo, _texto

# Arquitecturas admitidas, con los pesos preentrenados que usa timm por defecto y su licencia.
# Antes de añadir otra hay que comprobar la licencia de sus pesos (no todos los de timm son libres).
ARQUITECTURAS = {
    "efficientnet_b0": "timm/efficientnet_b0.ra_in1k (ImageNet-1k, Apache-2.0)",
}
MODOS_PESOS = ("ninguno", "inv_raiz", "inverso")
PARTICIONES = ("train", "valid", "test")
# "Muy suaves": por encima de esto el brillo o el contraste ya cambian el aspecto de la lechuga.
MAX_BRILLO_CONTRASTE = 0.2
# "Rotación pequeña": los giros grandes ya los dan los giros de 90 grados.
MAX_ROTACION_GRADOS = 30
# Claves que alguien podría escribir para pedir un cambio de color que no se permite.
_COLOR_PROHIBIDO = ("saturacion", "saturación", "saturation", "tono", "hue", "matiz")

# Claves esperadas de cada sección. Cualquier otra es un error (casi seguro una errata).
_ESQUEMA = {
    "modelo": ("arquitectura", "preentrenado", "resolucion", "dropout"),
    "entrenamiento": (
        "epocas", "batch_size", "lr_extractor", "lr_cabezas", "decaimiento_pesos",
        "epocas_calentamiento", "factor_lr_final", "precision_mixta", "num_workers",
        "dispositivo",
    ),
    "perdida": ("pesos_clase",),
    "aumentos": (
        "escala_recorte", "proporcion_recorte", "volteo_horizontal", "volteo_vertical",
        "giros_90", "rotacion_grados", "brillo", "contraste",
    ),
    "datos": ("dataset", "fuente_propia", "sobremuestreo_propio", "reparto_propio"),
    "salida": ("carpeta",),
    "smoke": ("epocas", "recortes_train", "recortes_eval"),
}  # fmt: skip
_CLAVES_RAIZ = ("semilla", *_ESQUEMA)


@dataclass(frozen=True)
class Aumentos:
    """Aumentos de train. Solo geometría, más brillo y contraste opcionales (0 = desactivado)."""

    escala_recorte: tuple[float, float]  # fracción del área que conserva el recorte aleatorio
    proporcion_recorte: tuple[float, float]  # ancho/alto del recorte; (1, 1) = sin deformar
    volteo_horizontal: bool
    volteo_vertical: bool
    giros_90: bool
    rotacion_grados: float  # rotación aleatoria entre -grados y +grados (0 = sin rotación)
    brillo: float
    contraste: float


@dataclass(frozen=True)
class ConfigEtapa2:
    """Parámetros de un entrenamiento de la etapa 2, ya validados."""

    arquitectura: str
    preentrenado: bool
    resolucion: int  # lado de la entrada cuadrada, en píxeles
    dropout: float
    epocas: int
    batch_size: int
    lr_extractor: float
    lr_cabezas: float
    decaimiento_pesos: float
    epocas_calentamiento: int
    factor_lr_final: float
    precision_mixta: bool
    num_workers: int
    dispositivo: str
    pesos_clase: str
    aumentos: Aumentos
    semilla: int
    dataset: str  # carpeta de recortes.csv, relativa a la raíz del repositorio
    fuente_propia: str
    sobremuestreo_propio: int
    reparto_propio: tuple[float, float, float] | None  # train, valid, test; None = el del CSV
    salida: str
    smoke_epocas: int
    smoke_recortes_train: int
    smoke_recortes_eval: int

    def a_diccionario(self) -> dict:
        """La configuración con la misma forma que el YAML (para guardarla con la ejecución)."""
        aumentos = self.aumentos
        reparto = None
        if self.reparto_propio is not None:
            reparto = dict(zip(PARTICIONES, self.reparto_propio, strict=True))
        return {
            "modelo": {
                "arquitectura": self.arquitectura,
                "preentrenado": self.preentrenado,
                "resolucion": self.resolucion,
                "dropout": self.dropout,
            },
            "entrenamiento": {
                "epocas": self.epocas,
                "batch_size": self.batch_size,
                "lr_extractor": self.lr_extractor,
                "lr_cabezas": self.lr_cabezas,
                "decaimiento_pesos": self.decaimiento_pesos,
                "epocas_calentamiento": self.epocas_calentamiento,
                "factor_lr_final": self.factor_lr_final,
                "precision_mixta": self.precision_mixta,
                "num_workers": self.num_workers,
                "dispositivo": self.dispositivo,
            },
            "perdida": {"pesos_clase": self.pesos_clase},
            "aumentos": {
                "escala_recorte": list(aumentos.escala_recorte),
                "proporcion_recorte": list(aumentos.proporcion_recorte),
                "volteo_horizontal": aumentos.volteo_horizontal,
                "volteo_vertical": aumentos.volteo_vertical,
                "giros_90": aumentos.giros_90,
                "rotacion_grados": aumentos.rotacion_grados,
                "brillo": aumentos.brillo,
                "contraste": aumentos.contraste,
            },
            "semilla": self.semilla,
            "datos": {
                "dataset": self.dataset,
                "fuente_propia": self.fuente_propia,
                "sobremuestreo_propio": self.sobremuestreo_propio,
                "reparto_propio": reparto,
            },
            "salida": {"carpeta": self.salida},
            "smoke": {
                "epocas": self.smoke_epocas,
                "recortes_train": self.smoke_recortes_train,
                "recortes_eval": self.smoke_recortes_eval,
            },
        }


def _comprobar_claves(config: dict) -> None:
    """Falta una sección o clave, o hay una desconocida -> ``ValueError`` que la nombra."""
    for clave in _CLAVES_RAIZ:
        if clave not in config:
            raise ValueError(f"falta '{clave}' en la configuración")
    desconocidas = [str(c) for c in config if c not in _CLAVES_RAIZ]
    for seccion, claves in _ESQUEMA.items():
        datos = config[seccion]
        if not isinstance(datos, dict):
            raise ValueError(f"'{seccion}' debe ser un bloque con claves (llegó {datos!r})")
        for clave in claves:
            if clave not in datos:
                raise ValueError(f"falta '{seccion}.{clave}' en la configuración")
        desconocidas += [f"{seccion}.{c}" for c in datos if c not in claves]
    for clave in desconocidas:
        if clave.startswith("aumentos.") and clave.split(".", 1)[1].lower() in _COLOR_PROHIBIDO:
            raise ValueError(
                f"{clave}: no se permiten cambios de tono ni de saturación (falsean el grupo "
                "color); como mucho brillo y contraste suaves"
            )
    if desconocidas:
        raise ValueError(f"claves desconocidas en la configuración: {', '.join(desconocidas)}")


def _arquitectura(valor, clave: str) -> str:
    if valor not in ARQUITECTURAS:
        raise ValueError(
            f"{clave}: arquitectura {valor!r} no admitida; válidas (licencia comprobada): "
            f"{', '.join(ARQUITECTURAS)}"
        )
    return valor


def _par(valor, clave: str, maximo: float) -> tuple[float, float]:
    """Una lista ``[a, b]`` con 0 < a <= b <= maximo."""
    if not isinstance(valor, list) or len(valor) != 2:
        raise ValueError(f"{clave}: debe ser una lista [mínimo, máximo] (llegó {valor!r})")
    a = float(_numero(valor[0], clave, 0, maximo, abierto_inf=True))
    b = float(_numero(valor[1], clave, 0, maximo, abierto_inf=True))
    if a > b:
        raise ValueError(f"{clave}: el mínimo no puede ser mayor que el máximo (llegó {valor!r})")
    return a, b


def _reparto(valor, clave: str) -> tuple[float, float, float] | None:
    """``null`` o ``{train, valid, test}`` con fracciones positivas que suman 1."""
    if valor is None:
        return None
    if not isinstance(valor, dict) or set(valor) != set(PARTICIONES):
        raise ValueError(
            f"{clave}: debe ser null o un bloque con train, valid y test (llegó {valor!r})"
        )
    fracciones = tuple(valor[p] for p in PARTICIONES)
    if any(isinstance(f, bool) or not isinstance(f, int | float) for f in fracciones):
        raise ValueError(f"{clave}: las fracciones deben ser números (llegó {valor!r})")
    try:
        _validar_fracciones(fracciones)
    except ValueError as error:
        raise ValueError(f"{clave}: {error}") from error
    return tuple(float(f) for f in fracciones)


def cargar_config_etapa2(ruta: str | Path) -> ConfigEtapa2:
    """Lee y valida un YAML de entrenamiento de la etapa 2.

    En lenguaje llano: abre el YAML, comprueba que están todas las secciones y claves (y ninguna
    de más), que cada valor tiene el tipo y el rango correctos (por ejemplo, el brillo y el
    contraste como mucho 0,2 y la rotación como mucho 30 grados), que la arquitectura es una de
    licencia comprobada y que las rutas son relativas, y lo devuelve como un objeto. Si algo
    falla, el error dice qué clave es (por ejemplo ``aumentos.brillo``).

    Errores: ``FileNotFoundError`` si el archivo no existe; ``ValueError`` si no es válido.
    """
    config = load_config(ruta)
    _comprobar_claves(config)
    modelo, entrenamiento = config["modelo"], config["entrenamiento"]
    aumentos, datos, smoke = config["aumentos"], config["datos"], config["smoke"]
    return ConfigEtapa2(
        arquitectura=_arquitectura(modelo["arquitectura"], "modelo.arquitectura"),
        preentrenado=_booleano(modelo["preentrenado"], "modelo.preentrenado"),
        resolucion=_entero(modelo["resolucion"], "modelo.resolucion", 32),
        dropout=float(_numero(modelo["dropout"], "modelo.dropout", 0, 1, abierto_sup=True)),
        epocas=_entero(entrenamiento["epocas"], "entrenamiento.epocas", 1),
        batch_size=_entero(entrenamiento["batch_size"], "entrenamiento.batch_size", 1),
        lr_extractor=_positivo(entrenamiento["lr_extractor"], "entrenamiento.lr_extractor"),
        lr_cabezas=_positivo(entrenamiento["lr_cabezas"], "entrenamiento.lr_cabezas"),
        decaimiento_pesos=float(_numero(entrenamiento["decaimiento_pesos"],
                                        "entrenamiento.decaimiento_pesos", 0, 1)),  # fmt: skip
        epocas_calentamiento=_entero(entrenamiento["epocas_calentamiento"],
                                     "entrenamiento.epocas_calentamiento", 0),  # fmt: skip
        factor_lr_final=float(_numero(entrenamiento["factor_lr_final"],
                                      "entrenamiento.factor_lr_final", 0, 1)),  # fmt: skip
        precision_mixta=_booleano(entrenamiento["precision_mixta"],
                                  "entrenamiento.precision_mixta"),  # fmt: skip
        num_workers=_entero(entrenamiento["num_workers"], "entrenamiento.num_workers", 0),
        dispositivo=_opcion(entrenamiento["dispositivo"], "entrenamiento.dispositivo",
                            DISPOSITIVOS),  # fmt: skip
        pesos_clase=_opcion(config["perdida"]["pesos_clase"], "perdida.pesos_clase", MODOS_PESOS),
        aumentos=Aumentos(
            escala_recorte=_par(aumentos["escala_recorte"], "aumentos.escala_recorte", 1),
            proporcion_recorte=_par(aumentos["proporcion_recorte"],
                                    "aumentos.proporcion_recorte", 10),  # fmt: skip
            volteo_horizontal=_booleano(aumentos["volteo_horizontal"],
                                        "aumentos.volteo_horizontal"),  # fmt: skip
            volteo_vertical=_booleano(aumentos["volteo_vertical"], "aumentos.volteo_vertical"),
            giros_90=_booleano(aumentos["giros_90"], "aumentos.giros_90"),
            rotacion_grados=float(_numero(aumentos["rotacion_grados"], "aumentos.rotacion_grados",
                                          0, MAX_ROTACION_GRADOS)),  # fmt: skip
            brillo=float(_numero(aumentos["brillo"], "aumentos.brillo", 0,
                                 MAX_BRILLO_CONTRASTE)),  # fmt: skip
            contraste=float(_numero(aumentos["contraste"], "aumentos.contraste", 0,
                                    MAX_BRILLO_CONTRASTE)),  # fmt: skip
        ),
        semilla=_entero(config["semilla"], "semilla", 0),
        dataset=_ruta_relativa(datos["dataset"], "datos.dataset"),
        fuente_propia=_texto(datos["fuente_propia"], "datos.fuente_propia"),
        sobremuestreo_propio=_entero(datos["sobremuestreo_propio"], "datos.sobremuestreo_propio",
                                     1),  # fmt: skip
        reparto_propio=_reparto(datos["reparto_propio"], "datos.reparto_propio"),
        salida=_ruta_relativa(config["salida"]["carpeta"], "salida.carpeta"),
        smoke_epocas=_entero(smoke["epocas"], "smoke.epocas", 1),
        smoke_recortes_train=_entero(smoke["recortes_train"], "smoke.recortes_train", 1),
        smoke_recortes_eval=_entero(smoke["recortes_eval"], "smoke.recortes_eval", 1),
    )
