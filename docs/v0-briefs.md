# Versión 0: briefs y prompts para Claude Code

Guardar este archivo en `docs/v0-briefs.md` del repositorio. Claude Code lo lee al empezar cada tarea, así que los prompts son cortos y el detalle vive aquí.

## Qué es la versión 0

Un programa que se ejecuta de extremo a extremo con modelos genéricos y sin datos propios. Valida la estructura del código, las pruebas y la medición de consumo. No evalúa calidad real: todavía no hay un modelo entrenado.

**Prueba de la versión 0 (criterio de hecho)**

1. `pytest -q` pasa y `ruff check .` no da avisos.
2. `python -m src.pipeline.run --input data/v0_test --out runs/v0_local --detector whole` genera, para cada imagen, un JSON y una imagen anotada.
3. Lo mismo con `--detector owlvit` en el portátil (CPU) y en Colab (T4).
4. Cada ejecución guarda tiempo por imagen, memoria y versiones de Python y PyTorch.
5. La nota `exp001` de Obsidian está completa, con el commit usado y los resultados medidos.
6. El repositorio lleva la etiqueta `v0`.

**Lo que la v0 no hace.** No detecta defectos ni emite ACCEPT: sin modelo de calidad, los seis grupos salen NO_VISIBLE y la salida global es REVIEW. No hay métricas de precisión. El detector genérico puede fallar con lechugas, y eso es esperable.

## Decisiones previas, con su valor por defecto

| Tema | Valor en la v0 | Pendiente de |
| --- | --- | --- |
| Detector | Imagen completa como una sola lechuga, y OWL-ViT opcional (Apache-2.0, zero-shot, [ficha](https://huggingface.co/google/owlvit-base-patch32)). COCO no tiene clase lechuga | Elegir el detector real en la v1 |
| Licencia del repositorio | Sin definir. La v0 no usa Ultralytics | Decisión AGPL-3.0 o Apache-2.0 |
| Salida global por etiqueta | Solo las 6 explícitas del manual. Las demás, REVIEW y marcadas como provisionales | Validación del tutor y el experto |
| Etiquetas YELLOW_BROWN | Dos valores: `YELLOW_BROWN_LIGHT` y `YELLOW_BROWN_MAJOR` | Confirmar con el tutor |
| Varias etiquetas por grupo | Un valor por grupo | Confirmar si hace falta multietiqueta |
| Imágenes de prueba | 3 a 5 fotos propias de una lechuga del súper (cenital, lateral y base), en `data/v0_test/`, sin subir a Git | Sirve también de captura piloto |

## Estructura objetivo

```text
src/
  config.py
  data/        schema.py, split.py
  evaluation/  aggregate.py
  pipeline/    types.py, run.py, draw.py,
               detectors/ (whole.py, owlvit.py), assessors/ (null.py)
  utils/       env.py, profiling.py
configs/       v0.yaml, salida_global.yaml
tests/         una carpeta de pruebas por módulo
```

## Reglas comunes a todas las tareas

- Una tarea por sesión: `/clear` al empezar y una rama `v0/<tarea>`.
- Modo plan primero (`Shift + Tab`): propone archivos, funciones y pruebas, sin editar, y espera la aprobación.
- En lógica determinista, primero las pruebas y después el código.
- Al terminar: pytest y ruff con su salida pegada, y la lista de archivos creados o modificados.
- No hace commit. El autor revisa con `git status` y `git --no-pager diff`.
- No añade dependencias sin pedirlo. No inventa datos, métricas ni versiones. No guarda imágenes ni pesos en el repositorio.
- Comentarios y docstrings en español, y cada función no trivial con una explicación en lenguaje llano.

**Prompt base** (se cambia solo el número de tarea):

```text
Lee CLAUDE.md y docs/v0-briefs.md. Trabaja solo en la TAREA N y aplica las reglas comunes del documento.
Entra primero en modo plan: propón archivos, funciones y pruebas, sin editar nada, y espera mi aprobación.
```

---

## TAREA 0: pruebas, linter y configuración

**Rama:** `v0/setup`

**Entrega**
- `pyproject.toml` con la configuración de pytest (`pythonpath = ["."]`, `testpaths = ["tests"]`, marca `slow` excluida por defecto) y de ruff (longitud de línea 100).
- `requirements.txt` con numpy, opencv-python, matplotlib, pandas, pyyaml y psutil. `requirements-dev.txt` con pytest y ruff. `requirements-detector.txt` con transformers. Sin `torch`.
- `src/config.py`: `load_config(ruta)` lee un YAML y devuelve un diccionario, con error claro si falta el archivo.
- `configs/v0.yaml` con el detector por defecto, el umbral y la consulta de texto.
- Una prueba mínima de `load_config`.
- Añadir a `CLAUDE.md` una sección "Desarrollo" con las reglas comunes.

**Criterio:** `pytest -q` pasa, `ruff check .` limpio y `python -c "from src.config import load_config"` funciona.

## TAREA 1: esquema de metadatos y partición por lechuga

**Rama:** `v0/datos`

**Entrega**
- `src/data/schema.py` con los seis grupos y sus valores permitidos, más `NO_VISIBLE`:
  - `corte`: CUT_OK, STEM_TOO_LONG, UNCLEAN_CUT, ROOT_REMAINS
  - `recorte`: TRIM_OK, UNDER_TRIMMED, OVER_TRIMMED
  - `hojas_ext`: OUTER_LEAVES_OK, OUTER_DAMAGE_LIGHT, OUTER_DAMAGE_MAJOR
  - `color`: COLOR_FRESH_OK, YELLOW_BROWN_LIGHT, YELLOW_BROWN_MAJOR, WILTING
  - `daño_fisico`: PHYSICAL_OK, TORN_LEAF, BRUISE_CRUSH
  - `daño_bio`: BIO_OK, ROT_DECAY, VISIBLE_PEST, PEST_DAMAGE, DISEASE_VISIBLE
- `validate_metadata(df)`: devuelve la lista de errores, sin lanzar excepción. Comprueba columnas presentes, valores válidos, `image_id` único y `fuente` y `licencia` no vacías.
- `src/data/split.py`: `group_split(df, group_col="lechuga_id", fracciones=(0.7, 0.15, 0.15), seed=42)`, determinista. Las filas sin `lechuga_id` forman cada una su propio grupo y emiten un aviso, porque en datasets públicos no se puede garantizar que no haya fugas.
- Columnas de `metadata.csv`: `image_id, fuente, licencia, etiqueta_original, lechuga_id, vista, sesion, corte, recorte, hojas_ext, color, daño_fisico, daño_bio, salida_global`.

**Pruebas (primero):** ninguna lechuga en dos particiones, mismo resultado con la misma semilla, proporciones aproximadas, valor inválido detectado, licencia vacía detectada, id vacío con aviso.

## TAREA 2: regla de agregación de la salida global

**Rama:** `v0/agregacion`

**Entrega**
- `configs/salida_global.yaml`: tabla etiqueta → nivel. Etiquetas OK → ACCEPT. Explícitas del manual: ROOT_REMAINS y ROT_DECAY → NON-CONFORMING; OUTER_DAMAGE_LIGHT, STEM_TOO_LONG y TORN_LEAF → REVIEW. Todas las demás con valor `PENDIENTE`, incluida VISIBLE_PEST (el manual es ambiguo).
- `src/evaluation/aggregate.py` con `aggregate(grupos, tabla)`, que devuelve el nivel y la lista de etiquetas provisionales.

**Reglas**
1. Cada etiqueta visible da su nivel. Una etiqueta `PENDIENTE` cuenta como REVIEW y se anota como provisional.
2. El resultado es el peor nivel de los grupos visibles.
3. ACCEPT solo si los seis grupos son visibles. Si alguno es NO_VISIBLE, el resultado no es mejor que REVIEW. El texto "máximo REVIEW" del manual se lee así y hay que confirmarlo con el tutor.
4. Sin ningún grupo visible, el resultado es REVIEW.
5. Una etiqueta desconocida lanza `ValueError` con un mensaje claro.

**Pruebas (primero):** todo OK da ACCEPT; un NO_VISIBLE da REVIEW; ROOT_REMAINS da NON-CONFORMING aunque el resto sea NO_VISIBLE; ningún grupo visible da REVIEW; etiqueta pendiente da REVIEW provisional; etiqueta desconocida lanza error; sustituir una etiqueta por otra peor nunca mejora el resultado.

## TAREA 3: pipeline de extremo a extremo

**Rama:** `v0/pipeline`

**Entrega** (implementada)
- `src/pipeline/types.py`: `Detection(box, score, label="lettuce")` (dataclass inmutable; `box` = `(x1, y1, x2, y2)` en píxeles) y los protocolos `Detector.detect(imagen)` y `Assessor.assess(imagen, deteccion)`. Las imágenes son arrays BGR de OpenCV.
- `detectors/whole.py`: `WholeDetector` devuelve la imagen completa, `(0, 0, ancho, alto)`, como una sola lechuga con puntuación 1.0.
- `detectors/__init__.py`: **registro de detectores** (`DETECTORS`, `crear_detector(nombre, cfg)`, `nombres_detectores()`). Un nombre desconocido lanza `ValueError` que lista los disponibles. Hoy solo existe `whole`; OWL-ViT se añade en la tarea 4 registrando su fábrica, que recibe la configuración de `configs/v0.yaml`.
- `assessors/null.py`: `NullAssessor` devuelve los seis grupos de `GROUPS` como `NO_VISIBLE`.
- `src/pipeline/imagenes.py`: `leer_imagen` y `guardar_imagen` con `cv2.imdecode` + `np.fromfile` y `cv2.imencode` + `tofile`, para que funcionen rutas con tildes y eñes en Windows. Extensiones admitidas: jpg, jpeg, png, webp y bmp.
- `src/pipeline/draw.py`: `dibujar` devuelve una copia con la caja y el nivel (más la puntuación) de cada lechuga, con color por nivel (verde, naranja, rojo). El texto se pasa a ASCII porque las fuentes de OpenCV no pintan tildes.
- `src/pipeline/run.py`: `python -m src.pipeline.run --input DIR [--out DIR] [--detector NOMBRE] [--config YAML] [--tabla YAML]`. Por cada imagen genera `<nombre>.json` y `<nombre>_anotada.jpg`. Reutiliza `load_config`, `GROUPS`/`NO_VISIBLE` de `schema.py` y `aggregate` con `configs/salida_global.yaml`; no duplica sus listas ni su lógica.
  - `main(argv)` se puede llamar desde las pruebas sin `subprocess`, y devuelve el código de salida: `0` si se procesó alguna imagen, `1` si ninguna y `2` si hay un error de uso (carpeta o configuración inexistentes, detector desconocido), con el motivo por la salida de errores.
  - Valores por defecto (calculados desde la raíz del repositorio, no desde la carpeta de trabajo): salida en `runs/v0` (ya ignorada por `.gitignore`), `configs/v0.yaml` y `configs/salida_global.yaml`. Si no se indica `--detector`, se usa el de `v0.yaml`.
  - Recorre la carpeta por orden alfabético, sin entrar en subcarpetas. Una extensión no admitida, una imagen ilegible o un nombre repetido con otra extensión (`foto.jpg` y `foto.png`) se saltan con un `UserWarning` y el proceso continúa.

**Cambios sobre el brief**
- `--detector` acepta los nombres del registro (hoy solo `whole`), no `whole|owlvit` fijos.
- `env` del JSON lleva solo `python` y `opencv`; la tarea 5 lo amplía con PyTorch y el dispositivo.
- El JSON se escribe en UTF-8 con `ensure_ascii=False` porque hay claves con eñe (`daño_fisico`, `daño_bio`).
- `label` de `Detection` no se vuelca al JSON (no figura en el esquema).
- Los tiempos son milisegundos con `time.perf_counter`, sin redondear. `total` cubre detección y evaluación, no la lectura ni la escritura de archivos.
- Se añadieron `--config`, `--tabla`, los códigos de salida y el aviso por nombre repetido; no estaban en el brief.

**Estructura del JSON** (los valores son de ejemplo, no resultados)

```json
{
  "image": "foto01.jpg",
  "lettuces": [
    {"box": [0, 0, 640, 480], "score": 1.0, "groups": {"corte": "NO_VISIBLE"}, "global": "REVIEW", "provisional_labels": []}
  ],
  "timing_ms": {"detect": 0.0, "assess": 0.0, "total": 0.0},
  "env": {"python": "...", "opencv": "..."}
}
```

En `groups` aparecen los seis grupos, en el orden de `GROUPS`; el ejemplo muestra solo uno.

**Criterios:** una imagen sin detecciones da `"lettuces": []` sin fallar; una imagen ilegible se salta con un aviso; las pruebas usan imágenes sintéticas creadas con numpy, nunca imágenes reales en el repositorio.

## TAREA 4: detector genérico opcional (OWL-ViT)

**Rama:** `v0/owlvit`

**Entrega** (implementada)
- `src/pipeline/detectors/owlvit.py`: `OwlViTDetector` con `OwlViTProcessor` y `OwlViTForObjectDetection` de `transformers`, modelo `google/owlvit-base-patch32`, registrado en el registro de detectores con el nombre `owlvit` (`crear_owlvit` es su fábrica).
  - **Importación perezosa:** `transformers` y `torch` solo se importan al crear el detector. Sin ellos, el error dice `pip install -r requirements-detector.txt`. En `main` ese error y los de red (`OSError`) acaban con código 2 y el motivo por la salida de errores.
  - **El modelo se carga una sola vez**, al crear el detector, no por imagen. El tiempo de carga queda en `tiempo_carga_s`, separado de la inferencia (que mide el pipeline en `timing_ms.detect`), y `run.py` lo muestra como `Carga del modelo: X s`. No se añade al JSON.
  - **Dispositivo:** `auto` usa `cuda` si hay GPU y `cpu` si no; `cuda` sin GPU es un error, no cae a CPU en silencio.
  - **Postprocesado de la librería:** se usa `OwlViTProcessor.post_process_grounded_object_detection(outputs, threshold, target_sizes)`, comprobado en `transformers` 5.19.0. En esa versión el procesador ya no tiene `post_process_object_detection` (sigue en `OwlViTImageProcessor`). `target_sizes` se pasa como `[(alto, ancho)]` de la imagen original; la librería devuelve las cajas en píxeles y puede devolver cajas fuera de la imagen.
  - **Postprocesado propio** (`postprocesar`, en numpy): descarta valores no finitos y puntuaciones que no superan el umbral → recorta a la imagen y redondea a píxeles → descarta cajas degeneradas (sin ancho o alto, también las que el recorte deja sin área) → NMS → ordena de mayor a menor puntuación → aplica `max_detecciones`. El tope va después del NMS para que las cajas repetidas no ocupen sitio.
- `src/pipeline/nms.py`: `nms(cajas, puntuaciones, iou_umbral)`, NMS voraz por IoU en numpy, sin dependencias nuevas. Suprime solo si el IoU es **mayor** que el umbral. Se comprueba contra `torchvision.ops.nms` en las pruebas (si torchvision está instalado).
- `configs/v0.yaml`, bloque `owlvit`: `modelo`, `consulta` (`a photo of a lettuce`), `umbral` (0.1), `max_detecciones` (10), `nms_iou` (0.5) y `dispositivo` (`auto`). Todos son obligatorios: el YAML es la única fuente de valores y el código no tiene valores por defecto escondidos. `max_detecciones` es un tope elegido para la v0, no un resultado.
- `run.py`: `--umbral` y `--consulta` sustituyen esos valores en una ejecución sin editar el YAML (solo valen para `owlvit`; con otro detector se ignoran con un aviso).
- Una prueba `@pytest.mark.slow` con el modelo real y una imagen sintética: solo comprueba que se ejecuta y que las cajas son válidas, no que encuentre una lechuga. El resto de pruebas usan un procesador y un modelo falsos y no descargan nada.

**Cambios sobre el brief**
- Umbral estricto: se conservan las cajas con puntuación **mayor** que el umbral, igual que hace la librería ("por encima del umbral").
- Se añadieron `modelo`, `max_detecciones`, `nms_iou` y `dispositivo` a la configuración, el NMS y las opciones de línea de comandos, que no figuraban en el brief.

**Avisos:** la primera ejecución descarga el modelo desde Hugging Face y requiere conexión. Es un modelo genérico: puede dar falsos positivos y no detectar lechugas. En la v0 solo se mide, no se mejora.

**Datos medidos en esta tarea** (portátil, CPU, `transformers` 5.19.0, PyTorch 2.14.1+cpu; orientativos, la medición rigurosa es la tarea 5):
- Tamaño del modelo: `model.safetensors` 612 983 940 bytes; la caché de Hugging Face del modelo ocupa 587 MiB en disco.
- Carga con la caché ya descargada: 11,03 s y 13,46 s en dos ejecuciones. La prueba `slow` completa, con la primera descarga, tardó 40 s.
- Inferencia con una imagen sintética de 480×640: 511 ms la primera llamada y entre 349 y 362 ms las cuatro siguientes. Sin detecciones sobre esa imagen sintética.
- No se ha probado con fotos reales de lechuga: no hay ninguna en el repositorio.

## TAREA 5: medición de entorno y consumo

**Rama:** `v0/medicion`

**Entrega**
- `src/utils/env.py`: devuelve versiones de Python, PyTorch (si está) y OpenCV, el sistema operativo y el dispositivo.
- `src/utils/profiling.py`: tiempo por imagen (mediana de N repeticiones tras una de calentamiento) y memoria residente con `psutil`. Se integra en el JSON del pipeline.

**Pruebas:** que `env` devuelva las claves esperadas y que la mediana se calcule bien con tiempos simulados.

## TAREA 6: notebook de Colab

**Rama:** `v0/colab`

**Entrega:** `notebooks/01_v0_colab.ipynb`, que solo orquesta. Monta Drive, clona el repositorio, instala `requirements.txt` y `requirements-detector.txt`, imprime `git rev-parse --short HEAD`, `sys.version` y `torch.__version__`, ejecuta el pipeline con los dos detectores sobre las imágenes de Drive y guarda los resultados en `MyDrive/tfg-lechugas/runs/v0`.

**Antes de hacer commit:** borrar las salidas (*Clear All Outputs*) y comprobar que no queda ninguna ruta ni dato personal.

---

## Nota exp001 para Obsidian

Crear `docs/experimentos/exp001.md` con la plantilla y esto:

- `estado: planificado` → `hecho` al terminar. `objetivo: validar el pipeline de la v0`. `dataset: fotos propias de prueba, sin etiquetas`. `modelo: OWL-ViT base patch32 y detector de imagen completa`.
- Apartado Resultados, una fila por ejecución:

| Detector | Dispositivo | Imágenes | Mediana (ms) | Memoria (MB) | Observación |
| --- | --- | --- | --- | --- | --- |
| whole | CPU local | | | | |
| owlvit | CPU local | | | | |
| owlvit | Colab T4 | | | | |

Rellenar solo con datos medidos. Si algo no se ejecuta, se deja vacío.

## Orden de trabajo y cierre

1. Tarea 0, luego 1 y 2 (se pueden hacer en cualquier orden), luego 3, 4, 5 y 6.
2. Tras cada tarea: revisar el diff, pasar pytest y ruff, hacer commit en la rama y fusionar en `main` (`git switch main`, `git merge v0/<tarea>`, `git push`).
3. Al terminar la prueba de la v0: completar `exp001`, actualizar el Estado del TFG, anotar el uso de IA en `docs/uso-ia.md` y ejecutar `git tag v0` y `git push --tags`.

## Fuera del alcance de la v0

Datos propios, entrenamiento, métricas de calidad, validación de la tabla de salida global, elección del modelo final y licencia del repositorio.
