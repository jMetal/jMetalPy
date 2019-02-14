import time
from typing import TypeVar, List

from distributed import as_completed, Client

from jmetal.algorithm.singleobjective.genetic_algorithm import GeneticAlgorithm
from jmetal.config import store
from jmetal.core.algorithm import DynamicAlgorithm, Algorithm
from jmetal.core.operator import Mutation, Crossover, Selection
from jmetal.core.problem import Problem, DynamicProblem
from jmetal.operator import RankingAndCrowdingDistanceSelection
from jmetal.util.comparator import DominanceComparator, Comparator
from jmetal.util.solution_list import Evaluator, Generator
from jmetal.util.termination_criterion import TerminationCriterion

S = TypeVar('S')
R = TypeVar('R')

"""
.. module:: NSGA-II
   :platform: Unix, Windows
   :synopsis: NSGA-II (Non-dominance Sorting Genetic Algorithm II) implementation.

.. moduleauthor:: Antonio J. Nebro <antonio@lcc.uma.es>, Antonio Benítez-Hidalgo <antonio.b@uma.es>
"""


class NSGAII(GeneticAlgorithm[S, R]):

    def __init__(self,
                 problem: Problem,
                 population_size: int,
                 offspring_population_size: int,
                 mutation: Mutation,
                 crossover: Crossover,
                 selection: Selection,
                 termination_criterion: TerminationCriterion,
                 population_generator: Generator = store.default_generator,
                 population_evaluator: Evaluator = store.default_evaluator,
                 dominance_comparator: Comparator = DominanceComparator()):
        """  NSGA-II implementation as described in

        * K. Deb, A. Pratap, S. Agarwal and T. Meyarivan, "A fast and elitist
          multiobjective genetic algorithm: NSGA-II," in IEEE Transactions on Evolutionary Computation,
          vol. 6, no. 2, pp. 182-197, Apr 2002. doi: 10.1109/4235.996017

        NSGA-II is a genetic algorithm (GA), i.e. it belongs to the evolutionary algorithms (EAs)
        family. The implementation of NSGA-II provided in jMetalPy follows the evolutionary
        algorithm template described in the algorithm module (:py:mod:`jmetal.core.algorithm`).

        .. note:: A steady-state version of this algorithm can be run by setting the offspring size to 1 and the mating pool size to 2.

        :param problem: The problem to solve.
        :param population_size: Size of the population.
        :param mutation: Mutation operator (see :py:mod:`jmetal.operator.mutation`).
        :param crossover: Crossover operator (see :py:mod:`jmetal.operator.crossover`).
        :param selection: Selection operator (see :py:mod:`jmetal.operator.selection`).
        """
        super(NSGAII, self).__init__(
            problem=problem,
            population_size=population_size,
            offspring_population_size=offspring_population_size,
            mutation=mutation,
            crossover=crossover,
            selection=selection,
            termination_criterion=termination_criterion,
            population_evaluator=population_evaluator,
            population_generator=population_generator
        )
        self.dominance_comparator = dominance_comparator

    def replacement(self, population: List[S], offspring_population: List[S]) -> List[List[S]]:
        """ This method joins the current and offspring populations to produce the population of the next generation
        by applying the ranking and crowding distance selection.

        :param population: Parent population.
        :param offspring_population: Offspring population.
        :return: New population after ranking and crowding distance selection is applied.
        """
        join_population = population + offspring_population

        return RankingAndCrowdingDistanceSelection(
            self.population_size, dominance_comparator=self.dominance_comparator
        ).execute(join_population)

    def get_result(self) -> R:
        return self.solutions

    def get_name(self) -> str:
        return 'NSGAII'


class DynamicNSGAII(NSGAII[S, R], DynamicAlgorithm):

    def __init__(self,
                 problem: DynamicProblem[S],
                 population_size: int,
                 offspring_population_size: int,
                 mutation: Mutation,
                 crossover: Crossover,
                 selection: Selection,
                 termination_criterion: TerminationCriterion,
                 population_generator: Generator = store.default_generator,
                 population_evaluator: Evaluator = store.default_evaluator,
                 dominance_comparator: DominanceComparator = DominanceComparator()):
        super(DynamicNSGAII, self).__init__(
            problem=problem,
            population_size=population_size,
            offspring_population_size=offspring_population_size,
            mutation=mutation,
            crossover=crossover,
            selection=selection,
            population_evaluator=population_evaluator,
            population_generator=population_generator,
            termination_criterion=termination_criterion,
            dominance_comparator=dominance_comparator)
        self.completed_iterations = 0
        self.start_computing_time = 0
        self.total_computing_time = 0

    def restart(self):
        self.solutions = self.evaluate(self.solutions)

    def update_progress(self):
        if self.problem.the_problem_has_changed():
            self.restart()
            self.problem.clear_changed()

        observable_data = self.get_observable_data()
        self.observable.notify_all(**observable_data)

        self.evaluations += self.offspring_population_size

    def stopping_condition_is_met(self):
        if self.termination_criterion.is_met:
            observable_data = self.get_observable_data()
            observable_data['TERMINATION_CRITERIA_IS_MET'] = True
            self.observable.notify_all(**observable_data)

            self.restart()
            self.init_progress()

            self.completed_iterations += 1


class DistributedNSGAII(Algorithm[S, R]):

    def __init__(self,
                 problem: Problem[S],
                 population_size: int,
                 mutation: Mutation[S],
                 crossover: Crossover[S, S],
                 selection: Selection[List[S], S],
                 termination_criterion: TerminationCriterion,
                 number_of_cores: int,
                 client: Client):
        super(DistributedNSGAII, self).__init__()
        self.problem = problem
        self.population_size = population_size
        self.mutation_operator = mutation
        self.crossover_operator = crossover
        self.selection_operator = selection

        self.termination_criterion = termination_criterion
        self.observable.register(termination_criterion)

        self.number_of_cores = number_of_cores
        self.client = client

    def create_initial_solutions(self) -> List[S]:
        return [self.problem.create_solution() for _ in range(self.number_of_cores)]

    def evaluate(self, solutions: List[S]) -> List[S]:
        return [self.client.submit(self.problem.evaluate, solution) for solution in solutions]

    def stopping_condition_is_met(self) -> bool:
        return self.termination_criterion.is_met

    def get_observable_data(self) -> dict:
        ctime = time.time() - self.start_computing_time
        return {'PROBLEM': self.problem, 'EVALUATIONS': self.evaluations, 'SOLUTIONS': self.get_result(),
                'COMPUTING_TIME': ctime}

    def init_progress(self) -> None:
        self.evaluations = self.number_of_cores

        observable_data = self.get_observable_data()
        self.observable.notify_all(**observable_data)

    def step(self) -> None:
        pass

    def update_progress(self):
        observable_data = self.get_observable_data()
        self.observable.notify_all(**observable_data)

    def run(self):
        """ Execute the algorithm. """
        self.start_computing_time = time.time()

        population_to_evaluate = self.create_initial_solutions()
        task_pool = as_completed(self.evaluate(population_to_evaluate))

        self.init_progress()

        auxiliar_population = []
        for future in task_pool:
            # The initial population is not full
            if len(auxiliar_population) < self.population_size:
                received_solution = future.result()
                auxiliar_population.append(received_solution)

                new_task = self.client.submit(self.problem.evaluate, self.problem.create_solution())
                task_pool.add(new_task)
            # Perform an algorithm step to create a new solution to be evaluated
            else:
                offspring_population = []

                if not self.stopping_condition_is_met():
                    offspring_population.append(future.result())

                    # Replacement
                    join_population = auxiliar_population + offspring_population
                    auxiliar_population = RankingAndCrowdingDistanceSelection(self.population_size).execute(
                        join_population)

                    # Selection
                    mating_population = []

                    for _ in range(2):
                        solution = self.selection_operator.execute(population_to_evaluate)
                        mating_population.append(solution)

                    # Reproduction and evaluation
                    new_task = self.client.submit(reproduction, mating_population, self.problem,
                                                  self.crossover_operator, self.mutation_operator)

                    task_pool.add(new_task)
                else:
                    for future in task_pool.futures:
                        future.cancel()
                    break

                self.evaluations += 1
                self.solutions = auxiliar_population

                self.update_progress()

        self.total_computing_time = time.time() - self.start_computing_time
        self.solutions = auxiliar_population

    def get_result(self) -> R:
        return self.solutions

    def get_name(self) -> str:
        return 'dNSGA-II'


def reproduction(mating_population: List[S], problem, crossover_operator, mutation_operator) -> S:
    offspring_pool = []
    for parents in zip(*[iter(mating_population)] * 2):
        offspring_pool.append(crossover_operator.execute(parents))

    offspring_population = []
    for pair in offspring_pool:
        for solution in pair:
            mutated_solution = mutation_operator.execute(solution)
            offspring_population.append(mutated_solution)

    return problem.evaluate(offspring_population[0])
