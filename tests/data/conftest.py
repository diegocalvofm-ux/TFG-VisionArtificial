"""Fixtures compartidas por las pruebas de src/data (datos sintéticos, sin imágenes)."""

import copy
import struct
from pathlib import Path

import pandas as pd
import pytest

RAIZ_REPO = Path(__file__).resolve().parents[2]

# Valores "OK" de cada grupo, escritos a mano a propósito: así la fábrica no depende
# del código que se está probando.
_VALORES_OK = {
    "corte": "CUT_OK",
    "recorte": "TRIM_OK",
    "hojas_ext": "OUTER_LEAVES_OK",
    "color": "COLOR_FRESH_OK",
    "daño_fisico": "PHYSICAL_OK",
    "daño_bio": "BIO_OK",
}
_VISTAS = ("cenital", "lateral_a", "lateral_b", "base")
# Columnas de contexto de captura: opcionales, así que en la tabla mínima válida van vacías.
_CONTEXTO_VACIO = {
    "variedad": "",
    "estado_seq": "",
    "horas_desde_corte": "",
    "conservacion": "",
    "temperatura_c": "",
    "intervencion": "",
    "dispositivo": "",
    "archivo_original": "",
}


@pytest.fixture
def metadata_valida():
    """Devuelve una función que fabrica un DataFrame de metadatos válido.

    Uso: ``df = metadata_valida(n_lechugas=10, fotos_por_lechuga=3)``. Cada lechuga
    (``L000``, ``L001``...) tiene varias fotos con ``image_id`` únicos (``img_0000``...).
    """

    def fabrica(n_lechugas: int = 3, fotos_por_lechuga: int = 2) -> pd.DataFrame:
        filas = []
        for i in range(n_lechugas):
            for j in range(fotos_por_lechuga):
                fila = {
                    "image_id": f"img_{len(filas):04d}",
                    "fuente": "propia",
                    "licencia": "propia",
                    "etiqueta_original": "",
                    "lechuga_id": f"L{i:03d}",
                    "vista": _VISTAS[j % len(_VISTAS)],
                    "sesion": "s1",
                    **_VALORES_OK,
                    "salida_global": "",
                    **_CONTEXTO_VACIO,
                }
                filas.append(fila)
        df = pd.DataFrame(filas)
        # Tipo ``object`` en las columnas de contexto: con el tipo texto de pandas 3 no se podría
        # asignar un número o un booleano a una celda, y las pruebas necesitan probar esos casos.
        return df.astype(dict.fromkeys(_CONTEXTO_VACIO, object))

    return fabrica


# --- ZIP COCO sintéticos (estilo Roboflow) para las pruebas del importador -----------------


def _jpeg_diminuto(semilla: int) -> bytes:
    """JPEG de 20x20 píxeles de un solo color (distinto según la semilla). Nunca es real."""
    import cv2
    import numpy as np

    imagen = np.full((20, 20, 3), (semilla * 7) % 256, dtype=np.uint8)
    correcto, datos = cv2.imencode(".jpg", imagen)
    assert correcto
    return datos.tobytes()


def _escribir_zip_coco(ruta, carpetas: dict, categorias: tuple[str, ...]) -> None:
    """Escribe un ZIP con carpetas train/valid/test y un _annotations.coco.json por carpeta.

    ``carpetas`` es ``{carpeta: [{"nombre": "a.jpg", "anotaciones": [(clase, poligonos)]}]}``.
    ``categorias`` se numera desde 0; como en Roboflow, la primera suele ser una categoría
    "padre" sin anotaciones.
    """
    import json
    import zipfile

    with zipfile.ZipFile(ruta, "w") as zf:
        for carpeta, imagenes in carpetas.items():
            coco = {
                "info": {},
                "licenses": [],
                "categories": [{"id": i, "name": n} for i, n in enumerate(categorias)],
                "images": [],
                "annotations": [],
            }
            for i, imagen in enumerate(imagenes):
                coco["images"].append(
                    {"id": i, "file_name": imagen["nombre"], "width": 20, "height": 20}
                )
                for clase, poligonos in imagen.get("anotaciones", []):
                    coco["annotations"].append(
                        {
                            "id": len(coco["annotations"]),
                            "image_id": i,
                            "category_id": categorias.index(clase),
                            "bbox": [0, 0, 1, 1],  # a propósito distinta de la real
                            "iscrowd": 0,
                            "area": 1,
                            "segmentation": poligonos,
                        }
                    )
                zf.writestr(f"{carpeta}/{imagen['nombre']}", _jpeg_diminuto(i + len(carpeta)))
            zf.writestr(f"{carpeta}/_annotations.coco.json", json.dumps(coco, ensure_ascii=False))


@pytest.fixture
def zip_coco_sintetico(tmp_path):
    """Devuelve una función que crea un ZIP COCO mínimo y devuelve su ruta."""

    def fabrica(
        carpetas: dict, categorias: tuple[str, ...] = ("padre", "lechuga", "kern"), nombre="x.zip"
    ):
        ruta = tmp_path / "zips" / nombre
        ruta.parent.mkdir(parents=True, exist_ok=True)
        _escribir_zip_coco(ruta, carpetas, categorias)
        return ruta

    return fabrica


@pytest.fixture
def fuente_sintetica(tmp_path):
    """Devuelve una función que crea ZIP + YAML de una fuente bajo ``tmp_path`` (la "raíz").

    Devuelve la ruta del YAML. El ``zip`` del YAML es relativo a ``tmp_path``, como en el repo.
    """
    import yaml

    def fabrica(
        nombre: str,
        carpetas: dict,
        clases: dict,
        *,
        categorias: tuple[str, ...] = ("padre", "lechuga", "kern"),
        papel: str = "train_only",
        fracciones: dict | None = None,
        grupo_regex: str | None = None,
        particion_temporal: dict | None = None,
        excluir: tuple[str, ...] = (),
        licencia: str = "MIT",
    ):
        ruta_zip = tmp_path / "zips" / f"{nombre}.zip"
        ruta_zip.parent.mkdir(parents=True, exist_ok=True)
        _escribir_zip_coco(ruta_zip, carpetas, categorias)
        config = {
            "fuente": nombre,
            "licencia": licencia,
            "zip": f"zips/{nombre}.zip",
            "clases": clases,
            "papel": papel,
        }
        if fracciones is not None:
            config["fracciones"] = fracciones
        if grupo_regex is not None:
            config["grupo_regex"] = grupo_regex
        if particion_temporal is not None:
            config["particion_temporal"] = particion_temporal
        if excluir:
            config["excluir_imagenes"] = list(excluir)
        ruta_yaml = tmp_path / "configs" / "datasets" / f"{nombre}.yaml"
        ruta_yaml.parent.mkdir(parents=True, exist_ok=True)
        ruta_yaml.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
        return ruta_yaml

    return fabrica


@pytest.fixture
def etapa_sintetica(tmp_path):
    """Devuelve una función que escribe ``configs/datasets/etapa1_v1.yaml`` bajo ``tmp_path``."""
    import yaml

    def fabrica(fuentes: list[str], semilla: int = 42, salida: str = "data/processed/etapa1_v1"):
        ruta = tmp_path / "configs" / "datasets" / "etapa1_v1.yaml"
        ruta.parent.mkdir(parents=True, exist_ok=True)
        contenido = {"fuentes": fuentes, "semilla": semilla, "salida": salida}
        ruta.write_text(yaml.safe_dump(contenido), encoding="utf-8")
        return ruta

    return fabrica


# --- Fotos propias: hoja de captura (xlsx), jpg diminutos y YAML de la ingesta ------------

# Cabeceras de la hoja de captura en el orden de la hoja real, escritas a mano a propósito.
CABECERAS_HOJA = (
    "Lechuga",
    "Variedad",
    "Qué le has hecho (estado)",
    "Fecha",
    "Hora",
    "Hojas de fuera dañadas (nº total)",
    "Hojas amarillas o marrones (nº)",
    "¿Se ve el daño en las fotos?",
    "Notas",
)
_VARIEDAD_HOJA = {"LI": "Iceberg", "LB": "Batavia"}


def _fila_hoja(
    lechuga: str,
    estado,
    *,
    variedad: str | None = None,
    danadas=0,
    amarillas=0,
    se_ve=None,
    notas=None,
    fecha="01/11/2026",
    hora="10:00",
) -> list:
    """Una fila de la hoja, en el orden de ``CABECERAS_HOJA``. ``None`` es una celda vacía.

    Si no se da la variedad, sale del prefijo de la lechuga (LI = Iceberg, LB = Batavia).
    """
    if variedad is None:
        variedad = _VARIEDAD_HOJA[lechuga[:2]]
    return [lechuga, variedad, estado, fecha, hora, danadas, amarillas, se_ve, notas]


def _escribir_hoja(
    ruta: Path,
    filas: list[list],
    *,
    hoja: str = "Captura",
    cabeceras: tuple[str, ...] = CABECERAS_HOJA,
    titulo: str | None = None,
    combinar: tuple[str, ...] = (),
) -> Path:
    """Escribe una hoja xlsx sintética: título opcional, cabeceras, filas y celdas combinadas.

    ``combinar`` son rangos de Excel (``"A2:A4"``); como en Excel, al combinar solo la primera
    celda conserva su valor.
    """
    import openpyxl

    libro = openpyxl.Workbook()
    hoja_activa = libro.active
    hoja_activa.title = hoja
    if titulo is not None:
        hoja_activa.append([titulo])
    hoja_activa.append(list(cabeceras))
    for fila in filas:
        hoja_activa.append(fila)
    for rango in combinar:
        hoja_activa.merge_cells(rango)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    libro.save(ruta)
    return ruta


def _segmento_exif(orientacion: int) -> bytes:
    """Segmento APP1 de un JPEG con un EXIF mínimo: solo la etiqueta de orientación (0x0112).

    En lenguaje llano: un EXIF es una pequeña tabla TIFF. Aquí se escribe a mano una tabla con
    una sola entrada (orientación, tipo SHORT, un valor), sin necesidad de Pillow.
    """
    entrada = struct.pack("<HHIHH", 0x0112, 3, 1, orientacion, 0)
    tabla = struct.pack("<H", 1) + entrada + struct.pack("<I", 0)
    carga = b"Exif\x00\x00" + b"II*\x00" + struct.pack("<I", 8) + tabla
    return b"\xff\xe1" + struct.pack(">H", len(carga) + 2) + carga


def _escribir_jpg(
    ruta: Path,
    ancho: int = 8,
    alto: int = 6,
    *,
    orientacion: int | None = None,
    comentario: bytes | None = None,
) -> Path:
    """Escribe un JPEG diminuto: mitad izquierda roja y mitad derecha azul. Nunca es real.

    Con ``orientacion`` se añade un EXIF con esa orientación y con ``comentario`` un segmento de
    comentario (COM): sirven para comprobar que la ingesta gira la foto y quita los metadatos.
    Los segmentos se insertan detrás del APP0 (JFIF) que escribe OpenCV.
    """
    import cv2
    import numpy as np

    imagen = np.zeros((alto, ancho, 3), dtype=np.uint8)
    imagen[:, : ancho // 2] = (0, 0, 255)  # rojo (BGR)
    imagen[:, ancho // 2 :] = (255, 0, 0)  # azul (BGR)
    correcto, datos = cv2.imencode(".jpg", imagen)
    assert correcto
    datos = datos.tobytes()
    extra = b""
    if orientacion is not None:
        extra += _segmento_exif(orientacion)
    if comentario is not None:
        extra += b"\xff\xfe" + struct.pack(">H", len(comentario) + 2) + comentario
    if extra:
        assert datos[2:4] == b"\xff\xe0"  # APP0 justo detrás del inicio de imagen
        fin_app0 = 4 + struct.unpack(">H", datos[4:6])[0]
        datos = datos[:fin_app0] + extra + datos[fin_app0:]
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_bytes(datos)
    return ruta


@pytest.fixture
def fila_hoja():
    """Devuelve la función que fabrica una fila de la hoja de captura (ver ``_fila_hoja``)."""
    return _fila_hoja


@pytest.fixture
def hoja_xlsx(tmp_path):
    """Devuelve una función que escribe una hoja xlsx sintética y devuelve su ruta."""

    def fabrica(filas: list[list], ruta: Path | None = None, **opciones) -> Path:
        return _escribir_hoja(ruta or tmp_path / "hoja.xlsx", filas, **opciones)

    return fabrica


@pytest.fixture
def jpg_sintetico():
    """Devuelve la función que escribe un JPEG diminuto (ver ``_escribir_jpg``)."""
    return _escribir_jpg


@pytest.fixture
def ingesta_sintetica(tmp_path):
    """Devuelve una función que prepara una ingesta completa bajo ``tmp_path`` (la "raíz").

    Uso: ``ruta_yaml = ingesta_sintetica(fotos, filas)``. Crea las fotos (nombres de
    ``fotos``) en ``raw/fotos``, la hoja en ``raw/hoja.xlsx`` y el YAML en
    ``configs/propio.yaml``. El YAML es una copia del real (``propio_piloto.yaml``) con las
    rutas cambiadas, así las pruebas usan las mismas reglas que la ingesta de verdad.
    ``cambios`` sustituye claves de primer nivel del YAML; ``excepciones`` es el texto de un
    CSV (``codificacion`` permite probar el BOM); ``hoja`` pasa opciones a la hoja.
    """
    import yaml

    def fabrica(
        fotos,
        filas: list[list],
        *,
        cambios: dict | None = None,
        excepciones: str | None = None,
        codificacion: str = "utf-8",
        hoja: dict | None = None,
    ) -> Path:
        for nombre in fotos:
            _escribir_jpg(tmp_path / "raw" / "fotos" / nombre)
        (tmp_path / "raw" / "fotos").mkdir(parents=True, exist_ok=True)
        _escribir_hoja(tmp_path / "raw" / "hoja.xlsx", filas, **(hoja or {}))

        with open(
            RAIZ_REPO / "configs" / "datasets" / "propio_piloto.yaml", encoding="utf-8"
        ) as archivo:
            config = copy.deepcopy(yaml.safe_load(archivo))
        config.update(fotos="raw/fotos", hoja="raw/hoja.xlsx", salida="processed/propio")
        if excepciones is not None:
            ruta_csv = tmp_path / "raw" / "excepciones.csv"
            ruta_csv.write_text(excepciones, encoding=codificacion)
            config["excepciones"] = "raw/excepciones.csv"
        config.update(cambios or {})

        ruta_yaml = tmp_path / "configs" / "propio.yaml"
        ruta_yaml.parent.mkdir(parents=True, exist_ok=True)
        # sort_keys=False: el orden de vistas y estados del YAML es el orden del informe.
        texto = yaml.safe_dump(config, allow_unicode=True, sort_keys=False)
        ruta_yaml.write_text(texto, encoding="utf-8")
        return ruta_yaml

    return fabrica


# --- Etapa 2: ZIP COCO con cajas, fotos propias y YAML del dataset de recortes ------------


def _imagen_suave(semilla: int, ancho: int = 96, alto: int = 72):
    """Imagen sintética con manchas de color al azar ampliadas (distinta para cada semilla).

    En lenguaje llano: una rejilla de 8x6 colores al azar ampliada con interpolación cúbica.
    Tiene estructura a gran escala, así su dHash es estable al comprimir en JPEG y dos semillas
    distintas dan huellas muy distintas (no salen duplicados por casualidad). Nunca es real.
    """
    import cv2
    import numpy as np

    pequena = np.random.default_rng(semilla).integers(0, 256, (6, 8, 3), dtype=np.uint8)
    return cv2.resize(pequena, (ancho, alto), interpolation=cv2.INTER_CUBIC)


def _escribir_zip_cajas(ruta: Path, carpetas: dict, categorias: tuple[str, ...]) -> Path:
    """Escribe un ZIP COCO (estilo Roboflow) con cajas en lugar de polígonos.

    ``carpetas`` es ``{carpeta: [imagen, ...]}`` y cada imagen un diccionario con ``nombre``,
    ``semilla``, ``cajas`` (lista de ``(clase, [x, y, w, h])``) y, opcionales, ``ancho``,
    ``alto`` (96x72 por defecto) y ``espejo`` (guarda la imagen volteada en horizontal). La misma
    semilla da la misma imagen: así se fabrican duplicados. ``categorias`` se numera desde 0.
    """
    import json
    import zipfile

    import cv2
    import numpy as np

    ruta.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ruta, "w") as zf:
        for carpeta, imagenes in carpetas.items():
            coco = {
                "info": {},
                "licenses": [],
                "categories": [{"id": i, "name": n} for i, n in enumerate(categorias)],
                "images": [],
                "annotations": [],
            }
            for i, imagen in enumerate(imagenes):
                ancho, alto = imagen.get("ancho", 96), imagen.get("alto", 72)
                pixeles = _imagen_suave(imagen["semilla"], ancho, alto)
                if imagen.get("espejo"):
                    pixeles = np.ascontiguousarray(np.fliplr(pixeles))
                correcto, datos = cv2.imencode(".jpg", pixeles)
                assert correcto
                zf.writestr(f"{carpeta}/{imagen['nombre']}", datos.tobytes())
                coco["images"].append(
                    {"id": i, "file_name": imagen["nombre"], "width": ancho, "height": alto}
                )
                for clase, caja in imagen.get("cajas", []):
                    coco["annotations"].append(
                        {
                            "id": len(coco["annotations"]),
                            "image_id": i,
                            "category_id": categorias.index(clase),
                            "bbox": list(caja),
                            "area": caja[2] * caja[3],
                            "segmentation": [],
                            "iscrowd": 0,
                        }
                    )
            zf.writestr(f"{carpeta}/_annotations.coco.json", json.dumps(coco, ensure_ascii=False))
    return ruta


@pytest.fixture
def imagen_suave():
    """Devuelve la función que fabrica una imagen sintética suave (ver ``_imagen_suave``)."""
    return _imagen_suave


@pytest.fixture
def fuente_etapa2(tmp_path):
    """Devuelve una función que crea el ZIP y el YAML descriptor de una fuente pública.

    Uso: ``fuente_etapa2("yssp", carpetas, categorias)``. El ZIP va a
    ``<tmp>/data/raw/<fuente>.zip`` y el YAML a ``<tmp>/configs/datasets/<fuente>.yaml`` (con
    ``fuente``, ``licencia`` y ``zip``, como los reales). Devuelve la ruta del ZIP.
    """
    import yaml

    def fabrica(nombre: str, carpetas: dict, categorias: tuple[str, ...], licencia: str = "MIT"):
        ruta_zip = tmp_path / "data" / "raw" / f"{nombre}.zip"
        _escribir_zip_cajas(ruta_zip, carpetas, categorias)
        descriptor = {
            "fuente": nombre,
            "nombre": f"dataset {nombre}",
            "licencia": licencia,
            "zip": f"data/raw/{nombre}.zip",
            "autor": None,
            "url": None,
            "fecha_descarga": "2026-11-16",
        }
        ruta_yaml = tmp_path / "configs" / "datasets" / f"{nombre}.yaml"
        ruta_yaml.parent.mkdir(parents=True, exist_ok=True)
        ruta_yaml.write_text(yaml.safe_dump(descriptor, allow_unicode=True), encoding="utf-8")
        return ruta_zip

    return fabrica


# Etiquetas de una foto propia de ejemplo: con defectos, para comprobar que se copian tal cual.
ETIQUETAS_PROPIAS = {
    "corte": "NO_VISIBLE",
    "recorte": "TRIM_OK",
    "hojas_ext": "OUTER_DAMAGE_LIGHT",
    "color": "YELLOW_BROWN_LIGHT",
    "daño_fisico": "TORN_LEAF",
    "daño_bio": "BIO_OK",
}
_VISTA_DE_LETRA = {"C": "cenital", "F": "lateral_a", "G": "lateral_b", "B": "base"}


@pytest.fixture
def etiquetas_propias():
    """Etiquetas que llevan en metadata.csv las fotos de ``propio_etapa2``."""
    return dict(ETIQUETAS_PROPIAS)


@pytest.fixture
def propio_etapa2(tmp_path):
    """Devuelve una función que prepara fotos propias ya ingeridas y su ``metadata.csv``.

    Uso: ``bloque = propio_etapa2(["LB-001_0_C", "LB-001_0_F"])``. Escribe cada foto en
    ``<tmp>/data/processed/propio_piloto/<image_id>.jpg`` (más un ``informe.json`` que debe
    ignorarse), las filas en ``<tmp>/data/metadata.csv`` (más una fila de otra fuente y otra de
    otra sesión, que no deben entrar) y unos pesos falsos en ``<tmp>/runs/pesos.pth``. Devuelve el
    bloque ``propio`` del YAML de la etapa 2. ``sin_fila`` y ``sin_foto`` dejan una foto sin su
    fila o una fila sin su foto.
    """
    import cv2
    import pandas as pd

    from src.data.schema import METADATA_COLUMNS

    def fabrica(
        image_ids: list[str],
        *,
        sin_fila: tuple[str, ...] = (),
        sin_foto: tuple[str, ...] = (),
        ancho: int = 160,
        alto: int = 120,
    ) -> dict:
        carpeta = tmp_path / "data" / "processed" / "propio_piloto"
        carpeta.mkdir(parents=True, exist_ok=True)
        (carpeta / "informe.json").write_text("{}", encoding="utf-8")
        filas = []
        for n, image_id in enumerate(image_ids):
            if image_id not in sin_foto:
                correcto, datos = cv2.imencode(".jpg", _imagen_suave(100 + n, ancho, alto))
                assert correcto
                (carpeta / f"{image_id}.jpg").write_bytes(datos.tobytes())
            if image_id in sin_fila:
                continue
            fila = dict.fromkeys(METADATA_COLUMNS, "")
            fila.update(ETIQUETAS_PROPIAS)
            fila.update(
                image_id=image_id,
                fuente="propio",
                licencia="propia",
                etiqueta_original="tal como viene",
                lechuga_id=image_id.split("_")[0],
                vista=_VISTA_DE_LETRA[image_id[-1]],
                sesion="piloto",
                salida_global="REVIEW",
                archivo_original=f"{image_id}.jpg",
            )
            filas.append(fila)
        extra = []
        if filas:  # filas que no deben entrar: otra fuente y otra sesión de la misma fuente
            extra = [
                dict(filas[0], image_id="svodk_0001", fuente="svodk"),
                dict(filas[0], image_id="LB-009_0_C", sesion="otra"),
            ]
        tabla = pd.DataFrame([*filas, *extra], columns=list(METADATA_COLUMNS))
        tabla.to_csv(tmp_path / "data" / "metadata.csv", index=False, encoding="utf-8")
        (tmp_path / "runs").mkdir(exist_ok=True)
        (tmp_path / "runs" / "pesos.pth").write_bytes(b"pesos falsos")
        return {
            "tipo": "propio",
            "imagenes": "data/processed/propio_piloto",
            "metadata": "data/metadata.csv",
            "sesion": "piloto",
            "detector": {"pesos": "runs/pesos.pth", "umbral": 0.5, "dispositivo": "cpu"},
            "control_lado_px": 64,
        }

    return fabrica


@pytest.fixture
def etapa2_sintetica(tmp_path):
    """Devuelve una función que escribe ``<tmp>/configs/datasets/etapa2_v1.yaml``.

    Uso: ``ruta = etapa2_sintetica({"yssp": bloque})``. ``recorte`` y ``duplicados`` sustituyen
    claves de esas secciones (``min_lado_px`` es 8 por defecto: las imágenes de prueba son
    pequeñas); ``cambios`` sustituye claves de primer nivel.
    """
    import yaml

    def fabrica(fuentes: dict, *, recorte=None, duplicados=None, **cambios) -> Path:
        config = {
            "salida": "data/processed/etapa2_v1",
            "semilla": 42,
            "recorte": {"margen": 0.10, "lado_px": 512, "calidad_jpeg": 95, "min_lado_px": 8},
            "duplicados": {
                "umbral_hamming": 4,
                "giros_y_espejos": True,
                "prioridad_particiones": ["test", "valid", "train"],
            },
            "fuentes": fuentes,
        }
        config["recorte"].update(recorte or {})
        config["duplicados"].update(duplicados or {})
        config.update(cambios)
        ruta = tmp_path / "configs" / "datasets" / "etapa2_v1.yaml"
        ruta.parent.mkdir(parents=True, exist_ok=True)
        texto = yaml.safe_dump(config, allow_unicode=True, sort_keys=False)
        ruta.write_text(texto, encoding="utf-8")
        return ruta

    return fabrica
