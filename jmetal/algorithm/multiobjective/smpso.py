from typing import TypeVar

from jmetal.component.archive import BoundedArchive
from jmetal.core.solution import FloatSolution

from jmetal.core.algorithm import ParticleSwarmOptimization
from jmetal.core.operator import Mutation
from jmetal.core.problem import Problem
from jmetal.util.comparator import DominanceComparator
from jmetal.util.observable import Observable, DefaultObservable

import numpy

R = TypeVar('R')


class SMPSO(ParticleSwarmOptimization[R]):
    def __init__(self,
                 problem: Problem[FloatSolution],
                 swarm_size: int,
                 max_evaluations: int,
                 mutation: Mutation[FloatSolution],
                 leaders: BoundedArchive[FloatSolution],
                 observable: Observable = DefaultObservable()):
        super(SMPSO, self).__init__()
        self.problem = problem
        self.swarm_size = swarm_size
        self.max_evaluations = max_evaluations
        self.mutation : Mutation[FloatSolution] = mutation
        self.leaders = leaders
        self.observable = observable

        self.evaluations = 0

        self.c1_min = 1.5
        self.c1_max = 2.5
        self.c2_min = 1.5
        self.c2_max = 2.5

        self.min_weight = 0.1
        self.max_weight = 0.1

        self.change_velocity1 = -1
        self.change_velocity2 = -1

        self.dominance_comparator = DominanceComparator()

        self.speed = numpy.empty((self.swarm_size, self.problem.number_of_variables), dtype=float)
        self.delta_max = numpy.empty(problem.number_of_variables)
        self.delta_min = numpy.empty(problem.number_of_variables)
        for i in range(problem.number_of_variables):
            self.delta_max[i] = (self.problem.upper_bound[i] - self.problem.lower_bound[i]) / 2.0

        self.delta_min = -1.0 * self.delta_max

    def init_progress(self) -> None :
        self.evaluations = self.swarm_size
        self.leaders.compute_density_estimator()

    def update_progress(self) -> None :
        self.evaluations += self.swarm_size
        self.leaders.compute_density_estimator()

    def is_stopping_condition_reached(self) -> bool:
        return self.evaluations >= self.max_evaluations

    def evaluate_swarm(self, swarm: List[FloatSolution]) -> List[FloatSolution]:
        pass

    def initialize_global_best(self, swarm: List[FloatSolution]) -> None:
        pass

    def initialize_particle_best(self, swarm: List[FloatSolution]) -> None:
        pass

    def initialize_velocity(self, swarm: List[FloatSolution]) -> None:
        pass

    def update_velocity(self, swarm: List[FloatSolution]) -> None:
        pass

    def update_position(self, swarm: List[FloatSolution]) -> None:
        pass

    def perturbation(self, swarm: List[FloatSolution]) -> None:
        pass

    def update_global_best(self, swarm: List[FloatSolution]) -> None:
        pass

    def update_particle_best(self, swarm: List[FloatSolution]) -> None:
        pass

    def get_result(self) -> R:
        pass
