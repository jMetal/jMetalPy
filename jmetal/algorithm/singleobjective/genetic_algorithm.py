from typing import TypeVar, List

from jmetal.component.evaluator import Evaluator
from jmetal.component.generator import Generator
from jmetal.core.algorithm import EvolutionaryAlgorithm
from jmetal.core.operator import Mutation, Crossover, Selection
from jmetal.core.problem import Problem
from jmetal.util.termination_criterion import TerminationCriterion

S = TypeVar('S')
R = TypeVar('R')

"""
.. module:: genetic_algorithm
   :platform: Unix, Windows
   :synopsis: Implementation of Genetic Algorithms.

.. moduleauthor:: Antonio J. Nebro <antonio@lcc.uma.es>, Antonio Benítez-Hidalgo <antonio.b@uma.es>
"""


class GeneticAlgorithm(EvolutionaryAlgorithm[S, R]):

    def __init__(self,
                 problem: Problem[S],
                 population_size: int,
                 offspring_population_size: int,
                 mating_pool_size: int,
                 mutation: Mutation,
                 crossover: Crossover,
                 selection: Selection,
                 termination_criterion: TerminationCriterion,
                 pop_generator: Generator = None,
                 pop_evaluator: Evaluator = None):
        """
        .. note:: A steady-state version of this algorithm can be run by setting the offspring size to 1 and the mating pool size to 2.
        """
        super(GeneticAlgorithm, self).__init__(
            problem=problem,
            population_size=population_size,
            offspring_population_size=offspring_population_size)
        self.mating_pool_size = mating_pool_size
        self.mutation_operator = mutation
        self.crossover_operator = crossover
        self.selection_operator = selection
        self.pop_generator = pop_generator
        self.pop_evaluator = pop_evaluator
        self.termination_criterion = termination_criterion

        self.observable.register(termination_criterion)

    def create_initial_solutions(self) -> List[S]:
        return [self.pop_generator.new(self.problem) for _ in range(self.population_size)]

    def evaluate(self, solution_list:List[S]):
        return self.pop_evaluator.evaluate(solution_list, self.problem)

    def stopping_condition_is_met(self) -> bool:
        return self.termination_criterion.is_met

    def selection(self, population: List[S]):
        mating_population = []

        for i in range(self.mating_pool_size):
            solution = self.selection_operator.execute(population)
            mating_population.append(solution)

        return mating_population

    def reproduction(self, population: List[S]) -> List[S]:
        number_of_parents_to_combine = self.crossover_operator.get_number_of_parents()

        if len(population) % number_of_parents_to_combine != 0:
            raise Exception('Wrong number of parents')

        offspring_population = []
        for i in range(0, self.offspring_population_size, number_of_parents_to_combine):
            parents = []
            for j in range(number_of_parents_to_combine):
                parents.append(population[i + j])

            offspring = self.crossover_operator.execute(parents)

            for solution in offspring:
                self.mutation_operator.execute(solution)
                offspring_population.append(solution)

        return offspring_population

    def replacement(self, population: List[S], offspring_population: List[S]) -> List[S]:
        population.sort(key=lambda s: s.objectives[0])

        offspring_population.append(population[0])
        offspring_population.append(population[1])

        offspring_population.sort(key=lambda s: s.objectives[0])

        offspring_population.pop()
        offspring_population.pop()

        return offspring_population

    def get_result(self) -> R:
        return self.population[0]

    def get_name(self) -> str:
        return 'Genetic algorithm'
