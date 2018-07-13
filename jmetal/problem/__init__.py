from .multiobjective.constrained import Srinivas, Tanaka
from .multiobjective.unconstrained import Kursawe, Fonseca, Schaffer, Viennet2
from .multiobjective.dtlz import DTLZ1, DTLZ2
from .multiobjective.zdt import ZDT1, ZDT2, ZDT3, ZDT4, ZDT6
from .singleobjective.unconstrained import OneMax, Sphere

__all__ = [
    'Srinivas', 'Tanaka',
    'Kursawe', 'Fonseca', 'Schaffer', 'Viennet2',
    'DTLZ1', 'DTLZ2',
    'ZDT1', 'ZDT2', 'ZDT3', 'ZDT4', 'ZDT6',
    'OneMax', 'Sphere'
]
