import random
import threading
from copy import copy
from math import sqrt
from typing import TypeVar, List, Optional

import numpy

from jmetal.config import store
from jmetal.core.algorithm import ParticleSwarmOptimization, DynamicAlgorithm
from jmetal.core.operator import Mutation
from jmetal.core.problem import FloatProblem, DynamicProblem
from jmetal.core.solution import FloatSolution
from jmetal.util.archive import BoundedArchive, ArchiveWithReferencePoint
from jmetal.util.comparator import DominanceComparator
from jmetal.util.evaluator import Evaluator
from jmetal.util.generator import Generator
from jmetal.util.termination_criterion import TerminationCriterion

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
                 mutation: Mutation,
                 leaders: Optional[BoundedArchive],
                 termination_criterion: TerminationCriterion = store.default_termination_criteria,
                 swarm_generator: Generator = store.default_generator,
                 swarm_evaluator: Evaluator = store.default_evaluator):
        """ This class implements the SMPSO algorithm as described in

        * SMPSO: A new PSO-based metaheuristic for multi-objective optimization
        * MCDM 2009. DOI: `<http://dx.doi.org/10.1109/MCDM.2009.4938830/>`_.

        The implementation of SMPSO provided in jMetalPy follows the algorithm template described in the algorithm
        templates section of the documentation.

        :param problem: The problem to solve.
        :param swarm_size: Size of the swarm.
        :param max_evaluations: Maximum number of evaluations/iterations.
        :param mutation: Mutation operator (see :py:mod:`jmetal.operator.mutation`).
        :param leaders: Archive for leaders.
        """
        super(SMPSO, self).__init__(
            problem=problem,
            swarm_size=swarm_size)
        self.swarm_generator = swarm_generator
        self.swarm_evaluator = swarm_evaluator
        self.termination_criterion = termination_criterion
        self.observable.register(termination_criterion)
        self.mutation_operator = mutation
        self.leaders = leaders

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
        self.delta_max, self.delta_min = numpy.empty(problem.number_of_variables), \
                                         numpy.empty(problem.number_of_variables)

    def create_initial_solutions(self) -> List[FloatSolution]:
        return [self.swarm_generator.new(self.problem) for _ in range(self.swarm_size)]

    def evaluate(self, solution_list: List[FloatSolution]):
        return self.swarm_evaluator.evaluate(solution_list, self.problem)

    def stopping_condition_is_met(self) -> bool:
        return self.termination_criterion.is_met

    def initialize_global_best(self, swarm: List[FloatSolution]) -> None:
        for particle in swarm:
            self.leaders.add(copy(particle))

    def initialize_particle_best(self, swarm: List[FloatSolution]) -> None:
        for particle in swarm:
            particle.attributes['local_best'] = copy(particle)

    def initialize_velocity(self, swarm: List[FloatSolution]) -> None:
        for i in range(self.problem.number_of_variables):
            self.delta_max[i] = (self.problem.upper_bound[i] - self.problem.lower_bound[i]) / 2.0

        self.delta_min = -1.0 * self.delta_max

    def update_velocity(self, swarm: List[FloatSolution]) -> None:
        for i in range(self.swarm_size):
            best_particle = copy(swarm[i].attributes['local_best'])
            best_global = self.select_global_best()

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
                        ((self.__inertia_weight(wmax)
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

    def update_global_best(self, swarm: List[FloatSolution]) -> None:
        for particle in swarm:
            self.leaders.add(copy(particle))

    def update_particle_best(self, swarm: List[FloatSolution]) -> None:
        for i in range(self.swarm_size):
            flag = self.dominance_comparator.compare(
                swarm[i],
                swarm[i].attributes['local_best'])
            if flag != 1:
                swarm[i].attributes['local_best'] = copy(swarm[i])

    def perturbation(self, swarm: List[FloatSolution]) -> None:
        for i in range(self.swarm_size):
            if (i % 6) == 0:
                self.mutation_operator.execute(swarm[i])

    def select_global_best(self) -> FloatSolution:
        leaders = self.leaders.solution_list

        if len(leaders) > 2:
            particles = random.sample(leaders, 2)

            if self.leaders.comparator.compare(particles[0], particles[1]) < 1:
                best_global = copy(particles[0])
            else:
                best_global = copy(particles[1])
        else:
            best_global = copy(self.leaders.solution_list[0])

        return best_global

    def __velocity_constriction(self, value: float, delta_max: [], delta_min: [], variable_index: int) -> float:
        result = value
        if value > delta_max[variable_index]:
            result = delta_max[variable_index]
        if value < delta_min[variable_index]:
            result = delta_min[variable_index]

        return result

    def __inertia_weight(self, wmax: float):
        return wmax

    def __constriction_coefficient(self, c1: float, c2: float) -> float:
        rho = c1 + c2
        if rho <= 4:
            result = 1.0
        else:
            result = 2.0 / (2.0 - rho - sqrt(pow(rho, 2.0) - 4.0 * rho))

        return result

    def init_progress(self) -> None:
        self.evaluations = self.swarm_size
        self.leaders.compute_density_estimator()

        self.initialize_velocity(self.solutions)
        self.initialize_particle_best(self.solutions)
        self.initialize_global_best(self.solutions)

    def update_progress(self) -> None:
        self.evaluations += self.swarm_size
        self.leaders.compute_density_estimator()

        observable_data = self.get_observable_data()
        observable_data['SOLUTIONS'] = self.leaders.solution_list
        self.observable.notify_all(**observable_data)

    def get_result(self) -> List[FloatSolution]:
        return self.leaders.solution_list

    def get_name(self) -> str:
        return 'SMPSO'


class DynamicSMPSO(SMPSO, DynamicAlgorithm):

    def __init__(self,
                 problem: DynamicProblem[FloatSolution],
                 swarm_size: int,
                 mutation: Mutation,
                 leaders: BoundedArchive,
                 termination_criterion: TerminationCriterion = store.default_termination_criteria,
                 swarm_generator: Generator = store.default_generator,
                 swarm_evaluator: Evaluator = store.default_evaluator):
        super(DynamicSMPSO, self).__init__(
            problem=problem,
            swarm_size=swarm_size,
            mutation=mutation,
            leaders=leaders,
            termination_criterion=termination_criterion,
            swarm_generator=swarm_generator,
            swarm_evaluator=swarm_evaluator)
        self.completed_iterations = 0

    def restart(self) -> None:
        self.solutions = self.create_initial_solutions()
        self.solutions = self.evaluate(self.solutions)

        self.leaders.__init__(self.leaders.maximum_size)

        self.initialize_velocity(self.solutions)
        self.initialize_particle_best(self.solutions)
        self.initialize_global_best(self.solutions)

        self.init_progress()

    def update_progress(self):
        if self.problem.the_problem_has_changed():
            self.restart()
            self.problem.clear_changed()

        observable_data = self.get_observable_data()
        self.observable.notify_all(**observable_data)

        self.evaluations += self.swarm_size
        self.leaders.compute_density_estimator()

    def stopping_condition_is_met(self):
        if self.termination_criterion.is_met:
            observable_data = self.get_observable_data()
            observable_data['termination_criterion_is_met'] = True
            self.observable.notify_all(**observable_data)

            self.restart()
            self.init_progress()
            self.completed_iterations += 1


class SMPSORP(SMPSO):

    def __init__(self,
                 problem: FloatProblem,
                 swarm_size: int,
                 mutation: Mutation,
                 reference_points: List[List[float]],
                 leaders: List[ArchiveWithReferencePoint],
                 termination_criterion: TerminationCriterion,
                 swarm_generator: Generator = store.default_generator,
                 swarm_evaluator: Evaluator = store.default_evaluator):
        """ This class implements the SMPSORP algorithm.

        :param problem: The problem to solve.
        :param swarm_size:
        :param mutation:
        :param leaders: List of bounded archives.
        :param swarm_evaluator: An evaluator object to evaluate the solutions in the population.
        """
        super(SMPSORP, self).__init__(
            problem=problem,
            swarm_size=swarm_size,
            mutation=mutation,
            leaders=None,
            swarm_generator=swarm_generator,
            swarm_evaluator=swarm_evaluator,
            termination_criterion=termination_criterion)
        self.leaders = leaders
        self.reference_points = reference_points
        self.lock = threading.Lock()

        thread = threading.Thread(target=_change_reference_point, args=(self,))
        thread.start()

    def initialize_global_best(self, swarm: List[FloatSolution]) -> None:
        for particle in swarm:
            for leader in self.leaders:
                leader.add(copy(particle))

    def update_global_best(self, swarm: List[FloatSolution]) -> None:
        for particle in swarm:
            for leader in self.leaders:
                leader.add(copy(particle))

    def select_global_best(self) -> FloatSolution:
        selected = False
        selected_swarm_index = 0

        while not selected:
            selected_swarm_index = random.randint(0, len(self.leaders) - 1)
            if len(self.leaders[selected_swarm_index].solution_list) != 0:
                selected = True

        leaders = self.leaders[selected_swarm_index].solution_list

        if len(leaders) > 2:
            particles = random.sample(leaders, 2)

            if self.leaders[selected_swarm_index].comparator.compare(particles[0], particles[1]) < 1:
                best_global = copy(particles[0])
            else:
                best_global = copy(particles[1])
        else:
            best_global = copy(self.leaders[selected_swarm_index].solution_list[0])

        return best_global

    def init_progress(self) -> None:
        self.evaluations = self.swarm_size

        for leader in self.leaders:
            leader.compute_density_estimator()

        self.initialize_velocity(self.solutions)
        self.initialize_particle_best(self.solutions)
        self.initialize_global_best(self.solutions)

    def update_progress(self) -> None:
        self.evaluations += self.swarm_size

        for leader in self.leaders:
            leader.filter()
            leader.compute_density_estimator()

        observable_data = self.get_observable_data()
        observable_data['REFERENCE_POINT'] = self.get_reference_point()
        self.observable.notify_all(**observable_data)

    def update_reference_point(self, new_reference_points: list):
        with self.lock:
            self.reference_points = new_reference_points

            for index, archive in enumerate(self.leaders):
                archive.update_reference_point(new_reference_points[index])

    def get_reference_point(self):
        with self.lock:
            return self.reference_points

    def get_result(self) -> List[FloatSolution]:
        result = []

        for leader in self.leaders:
            for solution in leader.solution_list:
                result.append(solution)

        return result

    def get_name(self) -> str:
        return 'SMPSO/RP'


def _change_reference_point(algorithm: SMPSORP):
    """ Auxiliar function to read new reference points from the keyboard for the SMPSO/RP algorithm
    """
    number_of_reference_points = len(algorithm.reference_points)
    number_of_objectives = algorithm.problem.number_of_objectives

    while True:
        print(f'Enter {number_of_reference_points}-points of dimension {number_of_objectives}: ')
        read = [float(x) for x in input().split()]

        # Update reference points
        reference_points = []
        for i in range(0, len(read), number_of_objectives):
            reference_points.append(read[i:i + number_of_objectives])

        algorithm.update_reference_point(reference_points)
