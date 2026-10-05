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