from copy import copy
import random
from math import sqrt
from typing import TypeVar, List

import numpy

from jmetal.component.archive import BoundedArchive
from jmetal.component.evaluator import Evaluator, SequentialEvaluator
from jmetal.core.algorithm import ParticleSwarmOptimization
from jmetal.core.operator import Mutation
from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution
from jmetal.util.comparator import DominanceComparator
from jmetal.util.observable import Observable, DefaultObservable

R = TypeVar('R')

"""
.. module:: SMPSO
   :platform: Unix, Windows
   :synopsis: Implementation of SMPSO.

.. moduleauthor:: Antonio Benítez-Hidalgo <antonio.b@uma.es>
"""


class SMPSO(ParticleSwarmOptimization):
    def __init__(self,
                 problem: FloatProblem,
                 swarm_size: int,
                 max_evaluations: int,
                 mutation: Mutation[FloatSolution],
                 leaders: BoundedArchive[FloatSolution],
                 observable: Observable = DefaultObservable(),
                 evaluator: Evaluator[FloatSolution] = SequentialEvaluator[FloatSolution]()):
        """ This class implements the SMPSO algorithm described in

        * SMPSO: A new PSO-based metaheuristic for multi-objective optimization
        * MCDM 2009. DOI: `<http://dx.doi.org/10.1109/MCDM.2009.4938830/>`_.

        The implementation of SMPSO provided in jMetalPy follows the
        algorithm template described in the algorithm templates section (:mod:`algorithm`) of the documentation.

        :param problem: The problem to solve.
        :param swarm_size:
        :param max_evaluations:
        :param mutation:
        :param leaders:
        :param observable:
        :param evaluator: An evaluator object to evaluate the solutions in the population.
        """
        super(SMPSO, self).__init__()
        self.problem = problem
        self.swarm_size = swarm_size
        self.max_evaluations = max_evaluations
        self.mutation = mutation
        self.leaders = leaders
        self.observable = observable
        self.evaluator = evaluator

        self.evaluations = 0

        self.c1_min = 1.5
        self.c1_max = 2.5
        self.c2_min = 1.5
        self.c2_max = 2.5
        self.r1_min = 0.0
        self.r1_max = 1.0
        self.r2_min = 0.0
        self.r2_max = 1.0
        self.min_weight = 0.1
        self.max_weight = 0.1

        self.change_velocity1 = -1
        self.change_velocity2 = -1

        self.dominance_comparator = DominanceComparator()

        self.speed = numpy.zeros((self.swarm_size, self.problem.number_of_variables), dtype=float)
        self.delta_max, self.delta_min = numpy.empty(problem.number_of_variables),\
                                         numpy.empty(problem.number_of_variables)
        for i in range(problem.number_of_variables):
            self.delta_max[i] = (self.problem.upper_bound[i] - self.problem.lower_bound[i]) / 2.0

        self.delta_min = -1.0 * self.delta_max

    def init_progress(self) -> None:
        self.evaluations = self.swarm_size
        self.leaders.compute_density_estimator()

    def update_progress(self) -> None:
        self.evaluations += self.swarm_size
        self.leaders.compute_density_estimator()

        observable_data = {'evaluations': self.evaluations,
                           'population': self.leaders.solution_list,
                           'computing time': self.get_current_computing_time()}

        self.observable.notify_all(**observable_data)

    def is_stopping_condition_reached(self) -> bool:
        return self.evaluations >= self.max_evaluations

    def create_initial_swarm(self) -> List[FloatSolution]:
        swarm = []
        for _ in range(self.swarm_size):
            swarm.append(self.problem.create_solution())
        return swarm

    def evaluate_swarm(self, swarm: List[FloatSolution]) -> List[FloatSolution]:
        return self.evaluator.evaluate(swarm, self.problem)

    def initialize_global_best(self, swarm: List[FloatSolution]) -> None:
        for particle in swarm:
            self.leaders.add(particle)

    def initialize_particle_best(self, swarm: List[FloatSolution]) -> None:
        for particle in swarm:
            particle.attributes["local_best"] = copy(particle)

    def initialize_velocity(self, swarm: List[FloatSolution]) -> None:
        pass  # Velocity initialized in the constructor

    def update_velocity(self, swarm: List[FloatSolution]) -> None:
        for i in range(self.swarm_size):
            best_particle = copy(swarm[i].attributes["local_best"])
            best_global = self.__select_global_best()

            r1 = round(random.uniform(self.r1_min, self.r1_max), 1)
            r2 = round(random.uniform(self.r2_min, self.r2_max), 1)
            c1 = round(random.uniform(self.c1_min, self.c1_max), 1)
            c2 = round(random.uniform(self.c2_min, self.c2_max), 1)
            wmax = self.max_weight
            wmin = self.min_weight

            for var in range(swarm[i].number_of_variables):
                self.speed[i][var] = \
                    self.__velocity_constriction(
                        self.__constriction_coefficient(c1, c2) *
                        ((self.__inertia_weight(self.evaluations, self.max_evaluations, wmax, wmin)
                         * self.speed[i][var])
                         + (c1 * r1 * (best_particle.variables[var] - swarm[i].variables[var]))
                         + (c2 * r2 * (best_global.variables[var] - swarm[i].variables[var]))
                         ),
                        self.delta_max, self.delta_min, var)

    def update_position(self, swarm: List[FloatSolution]) -> None:
        for i in range(self.swarm_size):
            particle = swarm[i]

            for j in range(particle.number_of_variables):
                particle.variables[j] += self.speed[i][j]

                if particle.variables[j] < self.problem.lower_bound[j]:
                    particle.variables[j] = self.problem.lower_bound[j]
                    self.speed[i][j] *= self.change_velocity1

                if particle.variables[j] > self.problem.upper_bound[j]:
                    particle.variables[j] = self.problem.upper_bound[j]
                    self.speed[i][j] *= self.change_velocity2

    def perturbation(self, swarm: List[FloatSolution]) -> None:
        for i in range(self.swarm_size):
            if (i % 6) == 0:
                self.mutation.execute(swarm[i])

    def update_global_best(self, swarm: List[FloatSolution]) -> None:
        for particle in swarm:
            self.leaders.add(copy(particle))

    def update_particle_best(self, swarm: List[FloatSolution]) -> None:
        for i in range(self.swarm_size):
            flag = self.dominance_comparator.compare(
                swarm[i],
                swarm[i].attributes["local_best"])
            if flag != 1:
                swarm[i].attributes["local_best"] = copy(swarm[i])

    def get_result(self) -> List[FloatSolution]:
        return self.leaders.solution_list

    def __select_global_best(self) -> FloatSolution:
        particles = random.sample(self.leaders.solution_list, 2)
        if self.leaders.get_comparator().compare(particles[0], particles[1]) < 1:
            best_global = copy(particles[0])
        else:
            best_global = copy(particles[1])

        return best_global

    def __velocity_constriction(self, value: float, delta_max: [], delta_min: [], variable_index: int) -> float:
        result = value
        if value > delta_max[variable_index]:
            result = delta_max[variable_index]
        if value < delta_min[variable_index]:
            result = delta_min[variable_index]

        return result

    def __inertia_weight(self, evaluations: int, max_evaluations: int, wmax: float, wmin: float):
        return wmax

    def __constriction_coefficient(self, c1: float, c2: float) -> float:
        rho = c1 + c2
        if rho <= 4:
            result = 1.0
        else:
            result = 2.0 / (2.0 - rho - sqrt(pow(rho, 2.0) - 4.0 * rho))

        return result
