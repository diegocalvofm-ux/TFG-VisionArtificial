"""Medición de tiempo (mediana de varias repeticiones) y de memoria residente."""

import statistics
import time

import psutil

_MIB = 1024**2  # bytes en un MB (en realidad MiB, como lo muestran los monitores de memoria)


def _entero(nombre: str, valor, minimo: int) -> None:
    """Lanza ValueError si ``valor`` no es un entero (ni booleano) mayor o igual que ``minimo``."""
    if isinstance(valor, bool) or not isinstance(valor, int) or valor < minimo:
        raise ValueError(
            f"{nombre} debe ser un entero mayor o igual que {minimo} (recibido: {valor!r})"
        )


def validar_repeticiones(repeticiones, calentamiento) -> None:
    """Comprueba que ``repeticiones`` sea un entero >= 1 y ``calentamiento`` un entero >= 0.

    Error: ``ValueError`` con el nombre del parámetro incorrecto.
    """
    _entero("repeticiones", repeticiones, 1)
    _entero("calentamiento", calentamiento, 0)


def mediana(valores) -> float:
    """Mediana de una lista de números: el valor del medio (con un número par, la media de los dos).

    En lenguaje llano: ordenados los valores de menor a mayor, es el que queda en el centro. A
    diferencia de la media, no se deja arrastrar por un valor extremo (una imagen que tardó
    mucho por un tirón del sistema).

    Error: ``ValueError`` si la lista está vacía.
    """
    valores = list(valores)
    if not valores:
        raise ValueError("No se puede calcular la mediana de una lista vacía")
    return float(statistics.median(valores))


def medir(func, repeticiones: int = 5, calentamiento: int = 1, reloj=time.perf_counter) -> float:
    """Mide cuánto tarda ``func()`` y devuelve la mediana en milisegundos.

    En lenguaje llano: la primera vez que se ejecuta algo suele ser más lento (cachés frías,
    inicializaciones). Por eso primero se ejecuta ``func`` ``calentamiento`` veces SIN medir y
    después ``repeticiones`` veces midiendo cada una con ``reloj``; el resultado es la mediana
    de esas medidas. Con ``repeticiones=1`` y ``calentamiento=0`` es una sola medida, como si
    no hubiera repetición.

    ``func`` no recibe argumentos (si necesita datos, se le pasan con una función anónima).
    ``reloj`` devuelve segundos y se puede sustituir (las pruebas usan uno simulado).

    Error: ``ValueError`` si ``repeticiones`` o ``calentamiento`` no son enteros válidos; en ese
    caso ``func`` no llega a ejecutarse.
    """
    validar_repeticiones(repeticiones, calentamiento)
    for _ in range(calentamiento):
        func()  # solo calienta: ni se mide ni cuenta
    duraciones = []
    for _ in range(repeticiones):
        antes = reloj()
        func()
        duraciones.append((reloj() - antes) * 1000)
    return mediana(duraciones)


def rss_mb() -> float:
    """Memoria residente actual del proceso, en MB (la que está en RAM ahora mismo).

    Es la memoria residente tras la inferencia (la del momento de llamar), no el pico: si el
    proceso llegó a usar más y luego liberó, aquí no se ve. Se mide con psutil.
    """
    return psutil.Process().memory_info().rss / _MIB
