import random
from typing import TypeVar, List

import numpy as np
from jmetal.component import RandomGenerator, DominanceComparator, NonDominatedSolutionListArchive

from jmetal.core.solution import FloatSolution

from jmetal.core.algorithm import EvolutionaryAlgorithm

from jmetal.algorithm.singleobjective.genetic_algorithm import GeneticAlgorithm
from jmetal.component.evaluator import Evaluator, SequentialEvaluator
from jmetal.component.generator import Generator
from jmetal.core.operator import Mutation, Crossover
from jmetal.core.problem import Problem
from jmetal.operator import DifferentialEvolutionCrossover, RankingAndCrowdingDistanceSelection
from jmetal.operator.selection import DifferentialEvolutionSelection
from jmetal.util.aggregative_function import AggregativeFunction
from jmetal.util.neighborhood import WeightNeighborhood
from jmetal.util.termination_criterion import TerminationCriterion

S = TypeVar('S')
R = List[S]


class GDE3(EvolutionaryAlgorithm[FloatSolution, FloatSolution]):

    def __init__(self,
                 problem: Problem,
                 population_size: int,
                 cr: float,
                 f: float,
                 termination_criterion: TerminationCriterion,
                 k: float = 0.5,
                 population_generator: Generator = RandomGenerator(),
                 evaluator: Evaluator = SequentialEvaluator(),
                 dominance_comparator = DominanceComparator()):
        """
        :param max_number_of_replaced_solutions: (eta in Zhang & Li paper).
        :param neighbourhood_selection_probability: Probability of mating with a solution in the neighborhood rather than the entire population (Delta in Zhang & Li paper).
        """
        super(GDE3, self).__init__(
            problem=problem,
            population_size=population_size,
            offspring_population_size=population_size)

        self.dominance_comparator = dominance_comparator
        self.selection_operator = DifferentialEvolutionSelection()
        self.crossover_operator = DifferentialEvolutionCrossover(cr, f, k)
        self.termination_criterion = termination_criterion
        self.population_generator = population_generator
        self.evaluator = evaluator

        self.observable.register(termination_criterion)

    def selection(self, population: List[FloatSolution]) -> List[FloatSolution]:
        mating_pool = []
        for i in range(self.population_size):
            self.selection_operator.set_index_to_exclude(i)
            selected_solutions: List[FloatSolution] = self.selection_operator.execute(self.solutions)
            mating_pool = mating_pool + selected_solutions

        return mating_pool

    def reproduction(self, mating_pool: List[S]) -> List[S]:
        offspring_population = []
        first_parent_index = 0
        for solution in self.solutions:
            self.crossover_operator.current_individual = solution
            parents = mating_pool[first_parent_index:first_parent_index+3]
            first_parent_index += 3

            offspring_population.append(self.crossover_operator.execute(parents)[0])

        return offspring_population

    def replacement(self, population: List[S], offspring_population: List[FloatSolution]) -> List[FloatSolution]:
        tmp_list = []

        for solution1, solution2 in zip(self.solutions, offspring_population):
            result = self.dominance_comparator.compare(solution1, solution2)
            if result == -1:
                tmp_list.append(solution1)
            elif result == 1:
                tmp_list.append(solution2)
            else:
                tmp_list.append(solution1)
                tmp_list.append(solution2)

        join_population = population + offspring_population
        return RankingAndCrowdingDistanceSelection(self.population_size, dominance_comparator=self.dominance_comparator).execute(join_population)

    def create_initial_solutions(self) -> List[FloatSolution]:
        return [self.population_generator.new(self.problem) for _ in range(self.population_size)]

    def evaluate(self, solution_list: List[FloatSolution]) -> List[FloatSolution]:
        return self.evaluator.evaluate(solution_list, self.problem)

    def stopping_condition_is_met(self) -> bool:
        return self.termination_criterion.is_met

    def get_result(self) -> List[FloatSolution]:
        archive = NonDominatedSolutionListArchive()
        for solution in self.solutions:
            archive.add(solution)
        return archive.solution_list

    def get_name(self) -> str:
        return 'GDE3'
