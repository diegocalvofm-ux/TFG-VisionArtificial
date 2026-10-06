# TFG: evaluación de calidad de lechugas por visión artificial

## Contexto
- Lechugas ya recolectadas, imágenes RGB desde cualquier vista.
- Etapa 1: localizar la lechuga. Etapa 2: evaluar 6 grupos y dar salida ACCEPT / REVIEW / NON-CONFORMING.
- Cada grupo vale OK, defecto concreto o NO_VISIBLE.

## Entorno
- Python 3.12. En local, PyTorch en CPU. El entrenamiento se hace en Colab.

## Estructura
- src/: toda la lógica (datos, modelos, métricas). Los notebooks solo la importan.
- notebooks/: orquestan el entrenamiento en Colab.
- configs/: parámetros de cada experimento.
- docs/: diario de experimentos y registro de uso de IA.
- data/: datos locales, fuera del repositorio (ignorada en .gitignore).
- scripts/: descarga y conversión de datasets.
- tests/: pruebas del código de src/.

## Reglas
- Nunca subir imágenes, datos, pesos ni secretos al repositorio.
- Partir los datos por lechuga (lechuga_id), nunca por imagen.
- Código en Python, comentarios y documentación en español.
- Explicar cada cambio importante: el autor debe poder defender el código.
- Antes de entrenar en Colab, probar en local con un subconjunto pequeño en CPU.

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

## Experimentos
- Cada experimento es una nota docs/experimentos/expNNN.md creada desde docs/plantillas/experimento.md.
- Al terminar un entrenamiento en Colab hay que rellenar commit, entorno, versiones de Python y PyTorch, métrica y resultado.
- No inventar nunca métricas ni resultados: si un dato no está, se deja vacío.