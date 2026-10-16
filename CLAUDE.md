# TFG: evaluación de calidad de lechugas por visión artificial

## Contexto
- Lechugas ya recolectadas, imágenes RGB desde cualquier vista.
- Etapa 1: localizar la lechuga. Etapa 2: evaluar 6 grupos y dar salida ACCEPT / REVIEW / NON-CONFORMING.
- Cada grupo vale OK, defecto concreto o NO_VISIBLE.

## Entorno
- Python 3.12. En local, PyTorch en CPU. El entrenamiento se hace en Colab.
- Windows con PowerShell: los comandos que se le den al autor deben ser de PowerShell. Entorno virtual `.venv` con Python 3.12.10.
- Colab usa otras versiones de Python y PyTorch: el código debe funcionar en ambos. No añadir `torch` ni `torchvision` a requirements.txt.
- Leer y escribir archivos con encoding="utf-8". Hay nombres con ñ (daño_fisico, daño_bio) y rutas con tildes.

## Estructura
- src/: toda la lógica (datos, modelos, métricas). Los notebooks solo la importan.
- notebooks/: orquestan el entrenamiento en Colab.
- configs/: parámetros de cada experimento.
- docs/: diario de experimentos y registro de uso de IA.
- data/: datos locales, fuera del repositorio (ignorada en .gitignore).
- scripts/: descarga y conversión de datasets.
- tests/: pruebas del código de src/.

- runs/: resultados de ejecuciones, ignorada en .gitignore.
- src/data/schema.py es la única fuente de los grupos y sus etiquetas válidas. configs/salida_global.yaml debe coincidir con ella y una prueba lo comprueba.
- .claude/commands/: comandos propios de Claude Code.

## Reglas
- Nunca subir imágenes, datos, pesos ni secretos al repositorio.
- Partir los datos por lechuga (lechuga_id), nunca por imagen.
- Código en Python, comentarios y documentación en español.
- Explicar cada cambio importante: el autor debe poder defender el código.
- Antes de entrenar en Colab, probar en local con un subconjunto pequeño en CPU.

- Las decisiones que dependen del tutor (tabla de salida global, codificación de YELLOW_BROWN, multietiqueta) se dejan como PENDIENTE o provisionales y se avisa. No se rellenan con criterios de otras fuentes.
- El repositorio es público: no escribir en el código ni en docs/ nombres, correos ni datos personales del autor, del tutor o del experto.
- Si falta un dato o una decisión depende de otra persona, preguntar en lugar de suponer.
- La fecha se consulta con un comando, nunca se supone.

## Desarrollo
Reglas comunes a todas las tareas (el detalle de cada tarea está en docs/v0-briefs.md).
- Una tarea por sesión, en una rama `v0/<tarea>`.
- Modo plan primero: proponer archivos, funciones y pruebas, sin editar, y esperar la aprobación.
- En lógica determinista, primero las pruebas y después el código.
- Comandos: `pytest -q` (las pruebas marcadas `slow` se saltan; se ejecutan con `pytest -m slow`) y `ruff check .`.
- Al terminar: pegar la salida de pytest y ruff y listar los archivos creados o modificados.
- No hacer commit: el autor revisa con `git status` y `git --no-pager diff`.
- No añadir dependencias sin pedirlo. No inventar datos, métricas ni versiones. No guardar imágenes ni pesos en el repositorio.
- Comentarios y docstrings en español, y cada función no trivial con una explicación en lenguaje llano.

- Al cerrar cada tarea, antes del resumen final: actualiza solo la sección de esa tarea en docs/v0-briefs.md para que describa lo implementado, incluidos los cambios sobre el brief, y añade una fila a docs/uso-ia.md (fecha del sistema, Claude Code, descripción de la tarea, partes tocadas) con la columna "revisado por el autor" en "No". No toques nada más de esos archivos.

- No hacer git add, commit ni push: el autor revisa con `git status` y `git --no-pager diff`.
- No tocar notebooks/00_prueba_GPU.ipynb, data/, runs/, .env ni .claude/settings.local.json.
- No ejecutar comandos destructivos (git reset --hard, git clean, borrar ramas o archivos, borrados recursivos) sin pedir permiso.
## Experimentos
- Cada experimento es una nota docs/experimentos/expNNN.md creada desde docs/plantillas/experimento.md.
- Al terminar un entrenamiento en Colab hay que rellenar commit, entorno, versiones de Python y PyTorch, métrica y resultado.
- No inventar nunca métricas ni resultados: si un dato no está, se deja vacío.
- Para registrar un experimento se usa el comando `/registrar-experimento` (script `python -m src.experimentos.registro`): rellena la nota con lo que se le da y lo medido en los JSON de runs/ (órdenes `nuevo`, `ejecucion` y `campos`). Las notas nunca llevan rutas absolutas ni el usuario del sistema.

