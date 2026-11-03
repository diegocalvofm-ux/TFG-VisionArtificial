"""Configuración de la evaluación de la localización (``configs/eval/localizacion.yaml``).

Todo parámetro sale de aquí: el código no lleva valores fijos. Un YAML mal escrito (falta una
clave, un valor fuera de rango o una clave con una errata) da un error que nombra la clave, en
lugar de evaluar con algo distinto de lo que se quería. Las rutas deben ser relativas a la raíz
del repositorio: así ninguna salida lleva una ruta absoluta (el repositorio es público).
"""

from dataclasses import dataclass
from pathlib import Path

from src.config import load_config
from src.experimentos.privacidad import contiene_ruta_absoluta
from src.train.config import DISPOSITIVOS as _DISPOSITIVOS_ENTRENAMIENTO
from src.train.config import _entero, _opcion, _texto

METODOS = ("owlvit", "rfdetr", "linea_base_hsv")
# En la evaluación se mide la latencia en CPU; ``auto`` no se admite para no medir otra cosa.
DISPOSITIVOS = tuple(d for d in _DISPOSITIVOS_ENTRENAMIENTO if d != "auto")

# Claves esperadas de cada sección. Cualquier otra es un error (casi seguro una errata).
_ESQUEMA = {
    "conjunto": ("carpeta", "particion", "clase"),
    "medicion": ("repeticiones", "calentamiento"),
    "owlvit": ("config_base", "consulta", "umbral", "umbral_ap", "max_detecciones"),
    "rfdetr": ("pesos", "umbral", "umbral_ap"),
    "linea_base_hsv": ("h", "s", "v", "apertura_px", "cierre_px", "area_min_frac"),
    "ejemplos": ("numero", "semilla"),
    "salida": ("carpeta",),
}
_CLAVES_RAIZ = ("metodos", "iou_min", "dispositivo", *_ESQUEMA)


@dataclass(frozen=True)
class ParametrosHSV:
    """Parámetros de la línea base clásica. ``h``, ``s`` y ``v`` son pares (mínimo, máximo)."""

    h: tuple[int, int]
    s: tuple[int, int]
    v: tuple[int, int]
    apertura_px: int
    cierre_px: int
    area_min_frac: float


@dataclass(frozen=True)
class ConfigLocalizacion:
    """Parámetros de una evaluación de la localización, ya validados.

    Los campos ``owlvit_consulta``, ``owlvit_umbral`` y ``owlvit_max_detecciones`` valen ``None``
    cuando el YAML dice ``null``: significa "usar el valor de ``configs/v0.yaml``".
    """

    conjunto_carpeta: str  # dataset COCO exportado, relativo a la raíz
    particion: str
    clase: str
    metodos: tuple[str, ...]
    iou_min: float
    dispositivo: str
    repeticiones: int
    calentamiento: int
    owlvit_config_base: str
    owlvit_consulta: str | None
    owlvit_umbral: float | None
    owlvit_umbral_ap: float
    owlvit_max_detecciones: int | None
    rfdetr_pesos: str
    rfdetr_umbral: float
    rfdetr_umbral_ap: float
    hsv: ParametrosHSV
    ejemplos_numero: int
    ejemplos_semilla: int
    salida: str  # carpeta de las ejecuciones (<salida>/<exp_id>/)

    def a_diccionario(self) -> dict:
        """La configuración con la misma forma que el YAML (para guardarla con la ejecución)."""
        return {
            "conjunto": {
                "carpeta": self.conjunto_carpeta,
                "particion": self.particion,
                "clase": self.clase,
            },
            "metodos": list(self.metodos),
            "iou_min": self.iou_min,
            "dispositivo": self.dispositivo,
            "medicion": {"repeticiones": self.repeticiones, "calentamiento": self.calentamiento},
            "owlvit": {
                "config_base": self.owlvit_config_base,
                "consulta": self.owlvit_consulta,
                "umbral": self.owlvit_umbral,
                "umbral_ap": self.owlvit_umbral_ap,
                "max_detecciones": self.owlvit_max_detecciones,
            },
            "rfdetr": {
                "pesos": self.rfdetr_pesos,
                "umbral": self.rfdetr_umbral,
                "umbral_ap": self.rfdetr_umbral_ap,
            },
            "linea_base_hsv": {
                "h": list(self.hsv.h),
                "s": list(self.hsv.s),
                "v": list(self.hsv.v),
                "apertura_px": self.hsv.apertura_px,
                "cierre_px": self.hsv.cierre_px,
                "area_min_frac": self.hsv.area_min_frac,
            },
            "ejemplos": {"numero": self.ejemplos_numero, "semilla": self.ejemplos_semilla},
            "salida": {"carpeta": self.salida},
        }


def _numero(
    valor, clave: str, minimo: float, maximo: float, *, abierto_inf=False, abierto_sup=False
):
    """Un número (no booleano) entre ``minimo`` y ``maximo``; los extremos pueden quedar fuera."""
    valido = isinstance(valor, int | float) and not isinstance(valor, bool)
    if valido:
        valido = (valor > minimo if abierto_inf else valor >= minimo) and (
            valor < maximo if abierto_sup else valor <= maximo
        )
    if not valido:
        izq, der = ("(" if abierto_inf else "["), (")" if abierto_sup else "]")
        raise ValueError(
            f"{clave}: debe ser un número en {izq}{minimo}, {maximo}{der} (llegó {valor!r})"
        )
    return valor


def _umbral(valor, clave: str) -> float:
    return float(_numero(valor, clave, 0, 1))


def _opcional(valor, validar):
    """``None`` si el YAML dice ``null``; si no, el valor ya validado."""
    return None if valor is None else validar(valor)


def _ruta_relativa(valor, clave: str) -> str:
    """Un texto no vacío que no sea una ruta absoluta (de Windows ni de Linux)."""
    texto = _texto(valor, clave)
    if Path(texto).is_absolute() or contiene_ruta_absoluta(texto):
        raise ValueError(f"{clave}: debe ser una ruta relativa a la raíz del repositorio")
    return texto


def _rango(valor, clave: str, maximo: int) -> tuple[int, int]:
    """Un par ``[mínimo, máximo]`` de enteros entre 0 y ``maximo``, con mínimo <= máximo."""
    if not isinstance(valor, list) or len(valor) != 2:
        raise ValueError(f"{clave}: debe ser una lista [mínimo, máximo] (llegó {valor!r})")
    minimo = _entero(valor[0], f"{clave}[0]", 0)
    superior = _entero(valor[1], f"{clave}[1]", 0)
    if superior > maximo or minimo > superior:
        raise ValueError(
            f"{clave}: debe cumplir 0 <= mínimo <= máximo <= {maximo} (llegó {valor!r})"
        )
    return minimo, superior


def _metodos(valor) -> tuple[str, ...]:
    if not isinstance(valor, list) or not valor:
        raise ValueError(f"metodos: debe ser una lista no vacía (llegó {valor!r})")
    desconocidos = [m for m in valor if m not in METODOS]
    if desconocidos:
        raise ValueError(
            f"metodos: método desconocido {', '.join(map(str, desconocidos))} "
            f"(válidos: {', '.join(METODOS)})"
        )
    if len(set(valor)) != len(valor):
        raise ValueError(f"metodos: hay métodos repetidos (llegó {valor!r})")
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


def cargar_config_localizacion(ruta: str | Path) -> ConfigLocalizacion:
    """Lee y valida el YAML de la evaluación de la localización.

    En lenguaje llano: abre el YAML, comprueba que están todas las secciones y claves (y ninguna
    de más), que cada valor tiene el tipo y el rango correctos, que las rutas son relativas y que
    el umbral operativo de cada método no es menor que el que se usa para calcular el AP. Si algo
    falla, el error dice qué clave es (por ejemplo ``rfdetr.umbral``).

    Errores: ``FileNotFoundError`` si el archivo no existe; ``ValueError`` si el contenido no es
    válido; ``yaml.YAMLError`` si el YAML está mal escrito.
    """
    config = load_config(ruta)
    _comprobar_claves(config)
    conjunto, medicion = config["conjunto"], config["medicion"]
    owlvit, rfdetr, hsv = config["owlvit"], config["rfdetr"], config["linea_base_hsv"]
    ejemplos, salida = config["ejemplos"], config["salida"]

    medidas = (
        _entero(medicion["repeticiones"], "medicion.repeticiones", 1),
        _entero(medicion["calentamiento"], "medicion.calentamiento", 0),
    )
    owl_ap = _umbral(owlvit["umbral_ap"], "owlvit.umbral_ap")
    owl_umbral = _opcional(owlvit["umbral"], lambda v: _umbral(v, "owlvit.umbral"))
    if owl_umbral is not None and owl_umbral < owl_ap:
        raise ValueError(
            f"owlvit.umbral ({owl_umbral}) no puede ser menor que owlvit.umbral_ap ({owl_ap})"
        )
    rf_ap = _umbral(rfdetr["umbral_ap"], "rfdetr.umbral_ap")
    rf_umbral = _umbral(rfdetr["umbral"], "rfdetr.umbral")
    if rf_umbral < rf_ap:
        raise ValueError(
            f"rfdetr.umbral ({rf_umbral}) no puede ser menor que rfdetr.umbral_ap ({rf_ap})"
        )

    return ConfigLocalizacion(
        conjunto_carpeta=_ruta_relativa(conjunto["carpeta"], "conjunto.carpeta"),
        particion=_texto(conjunto["particion"], "conjunto.particion"),
        clase=_texto(conjunto["clase"], "conjunto.clase"),
        metodos=_metodos(config["metodos"]),
        iou_min=float(_numero(config["iou_min"], "iou_min", 0, 1, abierto_inf=True)),
        dispositivo=_opcion(config["dispositivo"], "dispositivo", DISPOSITIVOS),
        repeticiones=medidas[0],
        calentamiento=medidas[1],
        owlvit_config_base=_ruta_relativa(owlvit["config_base"], "owlvit.config_base"),
        owlvit_consulta=_opcional(owlvit["consulta"], lambda v: _texto(v, "owlvit.consulta")),
        owlvit_umbral=owl_umbral,
        owlvit_umbral_ap=owl_ap,
        owlvit_max_detecciones=_opcional(
            owlvit["max_detecciones"], lambda v: _entero(v, "owlvit.max_detecciones", 1)
        ),
        rfdetr_pesos=_ruta_relativa(rfdetr["pesos"], "rfdetr.pesos"),
        rfdetr_umbral=rf_umbral,
        rfdetr_umbral_ap=rf_ap,
        hsv=ParametrosHSV(
            h=_rango(hsv["h"], "linea_base_hsv.h", 179),
            s=_rango(hsv["s"], "linea_base_hsv.s", 255),
            v=_rango(hsv["v"], "linea_base_hsv.v", 255),
            apertura_px=_entero(hsv["apertura_px"], "linea_base_hsv.apertura_px", 1),
            cierre_px=_entero(hsv["cierre_px"], "linea_base_hsv.cierre_px", 1),
            area_min_frac=float(
                _numero(
                    hsv["area_min_frac"], "linea_base_hsv.area_min_frac", 0, 1,
                    abierto_inf=True, abierto_sup=True,
                )
            ),
        ),
        ejemplos_numero=_entero(ejemplos["numero"], "ejemplos.numero", 0),
        ejemplos_semilla=_entero(ejemplos["semilla"], "ejemplos.semilla", 0),
        salida=_ruta_relativa(salida["carpeta"], "salida.carpeta"),
    )
