from typing import TypeVar

from jmetal.core.solution import FloatSolution

from jmetal.core.algorithm import ParticleSwarmOptimization
from jmetal.core.operator import Mutation
from jmetal.core.problem import Problem
from jmetal.util.observable import Observable, DefaultObservable

R = TypeVar('R')


class SMPSO(ParticleSwarmOptimization[FloatSolution, R]):
    def __init__(self,
                 problem: Problem[FloatSolution],
                 swarm_size: int,
                 max_evaluations: int,
                 mutation: Mutation[FloatSolution],
                 archive: BoundedArchive[FloatSolution],
                 observable: Observable = DefaultObservable()):
        super(SMPSO, self).__init__()
        self.problem = problem
        self.population_size = swarm_size
        self.max_evaluations = max_evaluations
        self.mutation_operator = mutation
        self.evaluations = 0
        self.observable = observable

        pass