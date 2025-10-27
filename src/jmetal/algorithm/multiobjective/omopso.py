import random
from copy import copy
from typing import List, Optional, TypeVar

import numpy

from jmetal.config import store
from jmetal.core.algorithm import ParticleSwarmOptimization
from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution
from jmetal.operator.mutation import NonUniformMutation
from jmetal.operator.mutation import UniformMutation
from jmetal.util.archive import BoundedArchive, NonDominatedSolutionsArchive
from jmetal.util.comparator import DominanceComparator, EpsilonDominanceComparator
from jmetal.util.evaluator import Evaluator
from jmetal.util.generator import Generator
from jmetal.util.termination_criterion import TerminationCriterion

R = TypeVar("R")

"""
.. module:: OMOPSO
   :platform: Unix, Windows
   :synopsis: Implementation of SMPSO.

.. moduleauthor:: Antonio J. Nebro <antonio@lcc.uma.es>
"""


class OMOPSO(ParticleSwarmOptimization):
    def __init__(
        self,
        problem: FloatProblem,
        swarm_size: int,
        uniform_mutation: UniformMutation,
        non_uniform_mutation: NonUniformMutation,
        leaders: Optional[BoundedArchive],
        epsilon: float,
        termination_criterion: TerminationCriterion,
        swarm_generator: Generator = store.default_generator,
        swarm_evaluator: Evaluator = store.default_evaluator,
    ):
        """This class implements the OMOPSO algorithm as described in

        todo Update this reference
        * SMPSO: A new PSO-based metaheuristic for multi-objective optimization

        The implementation of OMOPSO provided in jMetalPy follows the algorithm template described in the algorithm
        templates section of the documentation.

        :param problem: The problem to solve.
        :param swarm_size: Size of the swarm.
        :param leaders: Archive for leaders.
        """
        super(OMOPSO, self).__init__(problem=problem, swarm_size=swarm_size)
        self.swarm_generator = swarm_generator
        self.swarm_evaluator = swarm_evaluator

        self.termination_criterion = termination_criterion
        self.observable.register(termination_criterion)

        self.uniform_mutation = uniform_mutation
        self.non_uniform_mutation = non_uniform_mutation

        self.leaders = leaders

        self.epsilon = epsilon
        self.epsilon_archive = NonDominatedSolutionsArchive(EpsilonDominanceComparator(epsilon))

        self.c1_min = 1.5
        self.c1_max = 2.0
        self.c2_min = 1.5
        self.c2_max = 2.0
        self.r1_min = 0.0
        self.r1_max = 1.0
        self.r2_min = 0.0
        self.r2_max = 1.0
        self.weight_min = 0.1
        self.weight_max = 0.5
        self.change_velocity1 = -1
        self.change_velocity2 = -1

        self.dominance_comparator = DominanceComparator()

        self.speed = numpy.zeros((self.swarm_size, self.problem.number_of_variables()), dtype=float)

    def create_initial_solutions(self) -> List[FloatSolution]:
        return [self.swarm_generator.new(self.problem) for _ in range(self.swarm_size)]

    def evaluate(self, solution_list: List[FloatSolution]):
        return self.swarm_evaluator.evaluate(solution_list, self.problem)

    def stopping_condition_is_met(self) -> bool:
        return self.termination_criterion.is_met

    def initialize_global_best(self, swarm: List[FloatSolution]) -> None:
        for particle in swarm:
            if self.leaders.add(particle):
                self.epsilon_archive.add(copy(particle))

    def initialize_particle_best(self, swarm: List[FloatSolution]) -> None:
        for particle in swarm:
            particle.attributes["local_best"] = copy(particle)

    def initialize_velocity(self, swarm: List[FloatSolution]) -> None:
        for i in range(self.swarm_size):
            for j in range(self.problem.number_of_variables()):
                self.speed[i][j] = 0.0

    def update_velocity(self, swarm: List[FloatSolution]) -> None:
        for i in range(self.swarm_size):
            best_particle = copy(swarm[i].attributes["local_best"])
            best_global = self.select_global_best()

            r1 = round(random.uniform(self.r1_min, self.r1_max), 1)
            r2 = round(random.uniform(self.r2_min, self.r2_max), 1)
            c1 = round(random.uniform(self.c1_min, self.c1_max), 1)
            c2 = round(random.uniform(self.c2_min, self.c2_max), 1)
            w = round(random.uniform(self.weight_min, self.weight_max), 1)

            # Direct access to _variables for better performance
            particle_vars = swarm[i]._variables
            best_particle_vars = best_particle._variables
            best_global_vars = best_global._variables

            for var in range(len(particle_vars)):
                self.speed[i][var] = (
                    w * self.speed[i][var]
                    + (c1 * r1 * (best_particle_vars[var] - particle_vars[var]))
                    + (c2 * r2 * (best_global_vars[var] - particle_vars[var]))
                )

    def update_position(self, swarm: List[FloatSolution]) -> None:
        for i in range(self.swarm_size):
            particle = swarm[i]
            particle_vars = particle._variables  # Direct access to internal list
            lower_bounds = self.problem.lower_bound
            upper_bounds = self.problem.upper_bound

            for j in range(len(particle_vars)):
                # Update position
                particle_vars[j] += self.speed[i][j]

                # Apply bounds
                if particle_vars[j] < lower_bounds[j]:
                    particle_vars[j] = lower_bounds[j]
                    self.speed[i][j] *= self.change_velocity1
                elif particle_vars[j] > upper_bounds[j]:
                    particle_vars[j] = upper_bounds[j]
                    self.speed[i][j] *= self.change_velocity2

    def update_global_best(self, swarm: List[FloatSolution]) -> None:
        for particle in swarm:
            if self.leaders.add(copy(particle)):
                self.epsilon_archive.add(copy(particle))

    def update_particle_best(self, swarm: List[FloatSolution]) -> None:
        for i in range(self.swarm_size):
            flag = self.dominance_comparator.compare(swarm[i], swarm[i].attributes["local_best"])
            if flag != 1:
                swarm[i].attributes["local_best"] = copy(swarm[i])

    def perturbation(self, swarm: List[FloatSolution]) -> None:
        self.non_uniform_mutation.set_current_iteration(self.evaluations / self.swarm_size)
        for i in range(self.swarm_size):
            if (i % 3) == 0:
                self.non_uniform_mutation.execute(swarm[i])
            else:
                self.uniform_mutation.execute(swarm[i])

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

    def init_progress(self) -> None:
        self.evaluations = self.swarm_size
        self.leaders.compute_density_estimator()

        self.initialize_velocity(self.solutions)
        self.initialize_particle_best(self.solutions)
        self.initialize_global_best(self.solutions)

    def update_progress(self) -> None:
        self.evaluations += self.swarm_size
        self.leaders.compute_density_estimator()

        observable_data = self.observable_data()
        observable_data["SOLUTIONS"] = self.epsilon_archive.solution_list
        self.observable.notify_all(**observable_data)

    def result(self) -> List[FloatSolution]:
        return self.epsilon_archive.solution_list

    def get_name(self) -> str:
        return "OMOPSO"
