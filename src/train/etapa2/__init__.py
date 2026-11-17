"""Entrenamiento del clasificador de la etapa 2: seis grupos de calidad sobre recortes de lechuga.

Misma organización que el entrenamiento de la etapa 1 (``src/train/``):

- ``config.py``: lectura y validación de ``configs/train/etapa2_*.yaml``.
- ``datos.py``: ``recortes.csv``, reparto de la fuente propia, smoke y sobremuestreo.
- ``aumentos.py``: transformaciones de train y de evaluación y el ``Dataset`` de PyTorch.
- ``modelo.py``: extractor de timm con una cabeza por grupo, pesos y ``predecir``.
- ``perdida.py``: pesos por clase y pérdida que ignora los grupos NO_VISIBLE.
- ``metricas.py``: métricas por grupo solo con las muestras visibles y línea base mayoritaria.
- ``entrenar.py``: el bucle de entrenamiento, ``runs/<exp_id>/`` y la línea de comandos.

Uso: ``python -m src.train.etapa2 --config configs/train/etapa2_effb0.yaml --exp-id <id>``.
"""
