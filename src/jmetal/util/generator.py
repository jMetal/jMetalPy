import copy
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import numpy as np

from jmetal.core.problem import Problem
from jmetal.core.solution import Solution

R = TypeVar("R")

"""
.. module:: generator
   :platform: Unix, Windows
   :synopsis: Population generators implementation.

.. moduleauthor:: Antonio Benítez-Hidalgo <antonio.b@uma.es>
"""


class Generator(Generic[R], ABC):
    @abstractmethod
    def new(self, problem: Problem, rng: np.random.Generator | None = None) -> R:
        pass


class RandomGenerator(Generator):
    def new(self, problem: Problem, rng: np.random.Generator | None = None):
        return problem.create_solution(rng)


class InjectorGenerator(Generator):
    def __init__(self, solutions: list[Solution]):
        super().__init__()
        # Make copies of provided solutions using their __copy__ implementations
        self.population = [copy.copy(s) for s in solutions]

    def new(self, problem: Problem, rng: np.random.Generator | None = None):
        if len(self.population) > 0:
            # If we have more solutions to inject, return one from the list
            return self.population.pop()
        else:
            # Otherwise generate a new solution
            solution = problem.create_solution(rng)

        return solution
