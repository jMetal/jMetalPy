from .multiobjective.constrained import Srinivas, Tanaka
from .multiobjective.dtlz import *
from .multiobjective.lz09 import LZ09_F2
from .multiobjective.unconstrained import Fonseca, Kursawe, Schaffer, Viennet2
from .multiobjective.zdt import ZDT1, ZDT2, ZDT3, ZDT4, ZDT5, ZDT6
from .singleobjective.tsp import TSP
from .singleobjective.unconstrained import OneMax, Sphere

__all__ = [
    "Srinivas",
    "Tanaka",
    "Kursawe",
    "Fonseca",
    "Schaffer",
    "Viennet2",
    "DTLZ1",
    "DTLZ2",
    "DTLZ3",
    "DTLZ4",
    "DTLZ5",
    "DTLZ6",
    "DTLZ7",
    "ZDT1",
    "ZDT2",
    "ZDT3",
    "ZDT4",
    "ZDT5",
    "ZDT6",
    "LZ09_F2",
    "OneMax",
    "Sphere",
    "TSP",
]
