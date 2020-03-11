from typing import TypeVar, List

import numpy as np

from jmetal.algorithm.singleobjective.genetic_algorithm import GeneticAlgorithm
from jmetal.config import store
from jmetal.core.operator import Mutation, Crossover
from jmetal.core.problem import Problem
from jmetal.core.quality_indicator import EpsilonIndicator
from jmetal.operator import BinaryTournamentSelection
from jmetal.util.comparator import SolutionAttributeComparator
from jmetal.util.evaluator import Evaluator
from jmetal.util.generator import Generator
from jmetal.util.termination_criterion import TerminationCriterion

S = TypeVar('S')
R = TypeVar('R')


class IBEA(GeneticAlgorithm[S, R]):

    def __init__(self,
                 problem: Problem,
                 population_size: int,
                 offspring_population_size: int,
                 mutation: Mutation,
                 crossover: Crossover,
                 kappa: float,
                 termination_criterion: TerminationCriterion = store.default_termination_criteria,
                 population_generator: Generator = store.default_generator,
                 population_evaluator: Evaluator = store.default_evaluator):
        """  Epsilon IBEA implementation as described in

        * Zitzler, Eckart, and Simon Künzli. "Indicator-based selection in multiobjective search."
        In International Conference on Parallel Problem Solving from Nature, pp. 832-842. Springer,
        Berlin, Heidelberg, 2004.

        https://link.springer.com/chapter/10.1007/978-3-540-30217-9_84

        IBEA is a genetic algorithm (GA), i.e. it belongs to the evolutionary algorithms (EAs)
        family. The multi-objective search in IBEA is guided by a fitness associated to every solution,
        which is in turn controlled by a binary quality indicator. This implementation uses the so-called
        additive epsilon indicator, along with a binary tournament mating selector.

        :param problem: The problem to solve.
        :param population_size: Size of the population.
        :param mutation: Mutation operator (see :py:mod:`jmetal.operator.mutation`).
        :param crossover: Crossover operator (see :py:mod:`jmetal.operator.crossover`).
        :param kappa: Weight in the fitness computation.
        """

        selection = BinaryTournamentSelection(
            comparator=SolutionAttributeComparator(key='fitness', lowest_is_best=False))
        self.kappa = kappa

        super(IBEA, self).__init__(
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

    def compute_fitness_values(self, population: List[S], kappa: float) -> List[S]:
        for i in range(len(population)):
            population[i].attributes['fitness'] = 0

            for j in range(len(population)):
                if j != i:
                    population[i].attributes['fitness'] += -np.exp(
                        -EpsilonIndicator([population[i].objectives]).compute([population[j].objectives]) / self.kappa)
        return population

    def create_initial_solutions(self) -> List[S]:
        population = [self.population_generator.new(self.problem) for _ in range(self.population_size)]
        population = self.compute_fitness_values(population, self.kappa)

        return population

    def replacement(self, population: List[S], offspring_population: List[S]) -> List[List[S]]:
        join_population = population + offspring_population
        join_population_size = len(join_population)
        join_population = self.compute_fitness_values(join_population, self.kappa)

        while join_population_size > self.population_size:
            current_fitnesses = [individual.attributes['fitness'] for individual in join_population]
            index_worst = current_fitnesses.index(min(current_fitnesses))

            for i in range(join_population_size):
                join_population[i].attributes['fitness'] += np.exp(
                    - EpsilonIndicator([join_population[i].objectives]).compute([join_population[index_worst].objectives]) / self.kappa)

            join_population.pop(index_worst)
            join_population_size = join_population_size - 1

        return join_population

    def get_result(self) -> R:
        return self.solutions

    def get_name(self) -> str:
        return 'Epsilon-IBEA'
