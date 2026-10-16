"""Fixtures de las pruebas de src/utils: reloj, git y psutil falsos (sin tiempos ni git reales)."""

from types import SimpleNamespace

import pytest


class RelojFalso:
    """Un reloj que solo avanza cuando se le dice. Sustituye a ``time.perf_counter``."""

    def __init__(self):
        self.segundos = 0.0
        self.llamadas = 0

    def __call__(self) -> float:
        self.llamadas += 1
        return self.segundos

    def avanzar(self, milisegundos: float) -> None:
        self.segundos += milisegundos / 1000


def funcion_con_duraciones(reloj: RelojFalso, duraciones_ms):
    """Una función que, en cada llamada, "tarda" la siguiente duración de la lista."""
    pendientes = iter(duraciones_ms)
    llamadas = []

    def func():
        llamadas.append(1)
        reloj.avanzar(next(pendientes))

    func.llamadas = llamadas
    return func


class GitFalso:
    """Imita ``subprocess.run`` para git. Cada parámetro fija una respuesta."""

    def __init__(self, commit="abc1234", status="", codigo_rev=0, codigo_status=0, falta=False):
        self.commit, self.status = commit, status
        self.codigo_rev, self.codigo_status, self.falta = codigo_rev, codigo_status, falta
        self.llamadas: list[list[str]] = []

    def __call__(self, orden, **kwargs):
        self.llamadas.append(list(orden))
        if self.falta:
            raise FileNotFoundError("git")
        if "rev-parse" in orden:
            return SimpleNamespace(returncode=self.codigo_rev, stdout=f"{self.commit}\n", stderr="")
        return SimpleNamespace(returncode=self.codigo_status, stdout=self.status, stderr="")


@pytest.fixture
def reloj():
    return RelojFalso()


@pytest.fixture
def con_duraciones():
    """Devuelve ``funcion_con_duraciones`` para usarla desde las pruebas."""
    return funcion_con_duraciones


@pytest.fixture
def git_falso():
    return GitFalso
