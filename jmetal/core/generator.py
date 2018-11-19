from abc import ABCMeta, abstractmethod
from typing import TypeVar, Generic

from jmetal.core.problem import Problem

R = TypeVar('R')

"""
.. module:: generator
   :platform: Unix, Windows
   :synopsis: Population generators implementation.

.. moduleauthor:: Antonio Benítez-Hidalgo <antonio.b@uma.es>
"""


class Generator(Generic[R]):

    __metaclass__ = ABCMeta

    @abstractmethod
    def new(self, problem: Problem) -> R:
        pass
