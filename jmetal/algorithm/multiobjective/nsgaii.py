import time
from typing import TypeVar, List, Generic

from distributed import as_completed, Client

from jmetal.algorithm.singleobjective.genetic_algorithm import GeneticAlgorithm
from jmetal.config import store
from jmetal.core.algorithm import DynamicAlgorithm
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


class DistributedNSGAII(Generic[S, R]):

    def __init__(self,
                 problem: Problem[S],
                 population_size: int,
                 max_evaluations: int,
                 mutation: Mutation[S],
                 crossover: Crossover[S, S],
                 selection: Selection[List[S], S],
                 number_of_cores: int,
                 client: Client):
        self.problem = problem
        self.population_size = population_size
        self.max_evaluations = max_evaluations
        self.mutation_operator = mutation
        self.crossover_operator = crossover
        self.selection_operator = selection
        self.evaluations = 0
        self.start_computing_time = 0
        self.observable = store.default_observable

        self.population = None
        self.number_of_cores = number_of_cores
        self.client = client

    def update_progress(self, population):
        ctime = time.time() - self.start_computing_time
        observable_data = {'EVALUATIONS': self.evaluations, 'COMPUTING_TIME': ctime, 'SOLUTIONS': population,
                           'PROBLEM': self.problem}
        self.observable.notify_all(**observable_data)

    def create_initial_solutions(self) -> List[S]:
        return [self.problem.create_solution() for _ in range(self.population_size)]

    def run(self):
        population_to_evaluate = self.create_initial_solutions()

        self.start_computing_time = time.time()

        futures = []
        for solution in population_to_evaluate:
            futures.append(self.client.submit(self.problem.evaluate, solution))

        task_pool = as_completed(futures)
        population = []

        # MAIN LOOP
        for future in task_pool:
            self.evaluations += 1

            # The initial population is not full
            if len(population) < self.population_size:
                received_solution = future.result()
                population.append(received_solution)

                new_task = self.client.submit(self.problem.evaluate, self.problem.create_solution())
                task_pool.add(new_task)

            # Perform an algorithm step to create a new solution to be evaluated
            else:
                offspring_population = []
                if self.evaluations < self.max_evaluations:
                    offspring_population.append(future.result())

                    # Replacement
                    join_population = population + offspring_population
                    population = RankingAndCrowdingDistanceSelection(self.population_size).execute(join_population)

                    # Selection
                    mating_population = []
                    for _ in range(2):
                        solution = self.selection_operator.execute(population_to_evaluate)
                        mating_population.append(solution)

                    # Reproduction
                    offspring = self.crossover_operator.execute(mating_population)
                    self.mutation_operator.execute(offspring[0])

                    solution_to_evaluate = offspring[0]

                    # Evaluation
                    new_task = self.client.submit(self.problem.evaluate, solution_to_evaluate)
                    task_pool.add(new_task)
                else:
                    self.total_computing_time = time.time() - self.start_computing_time
                    print("TIME: " + str(self.total_computing_time))
                    for future in task_pool.futures:
                        future.cancel()

                    break

                self.evaluations += 1

                if self.evaluations % 10 == 0:
                    self.update_progress(population_to_evaluate)

        self.population = population

    def get_result(self) -> R:
        return self.population

    def get_name(self) -> str:
        return 'dNSGA-II'
