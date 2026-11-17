"""Permite lanzar el entrenamiento con ``python -m src.train.etapa2`` (ver ``entrenar.py``)."""

import sys

from src.train.etapa2.entrenar import main

if __name__ == "__main__":
    sys.exit(main())
