import random
from typing import TypeVar, List

import numpy as np

from jmetal.algorithm.singleobjective.genetic_algorithm import GeneticAlgorithm
from jmetal.component.evaluator import Evaluator
from jmetal.component.generator import Generator
from jmetal.core.operator import Mutation, Crossover
from jmetal.core.problem import Problem
from jmetal.util.aggregative_function import AggregativeFunction
from jmetal.util.neighborhood import WeightNeighborhood
from jmetal.util.termination_criterion import TerminationCriteria

S = TypeVar('S')
R = List[S]


class MOEAD(GeneticAlgorithm):

    def __init__(self,
                 problem: Problem,
                 population_size: int,
                 mutation: Mutation,
                 crossover: Crossover,
                 aggregative_function: AggregativeFunction,
                 neighbourhood: WeightNeighborhood,
                 neighbourhood_selection_probability: float,
                 max_number_of_replaced_solutions: int,
                 termination_criteria: TerminationCriteria,
                 pop_generator: Generator = None,
                 pop_evaluator: Evaluator = None):
        """
        :param max_number_of_replaced_solutions: (eta in Zhang & Li paper).
        :param neighbourhood_selection_probability: Probability of mating with a solution in the neighborhood rather than the entire population (Delta in Zhang & Li paper).
        """
        super(MOEAD, self).__init__(
            problem=problem,
            population_size=population_size,
            offspring_size=1,
            mating_pool_size=3,
            mutation=mutation,
            crossover=crossover,
            selection=None,
            pop_evaluator=pop_evaluator,
            population_generator=pop_generator,
            termination_criteria=termination_criteria
        )
        self.max_number_of_replaced_solutions = max_number_of_replaced_solutions
        self.fitness_function = aggregative_function
        self.neighbourhood = neighbourhood
        self.neighbourhood_selection_probability = neighbourhood_selection_probability

    @staticmethod
    def random_permutations(size):
        """ Picks position from 1 to size at random and increments when value is already picked.
        """
        permutations = np.empty((size,), dtype=np.int8)

        index = np.empty((size,))
        flag = np.empty((size,))

        for i in range(size):
            index[i] = i
            flag[i] = True

        counter = 0
        while counter < size:
            start = random.randint(0, size - 1)

            while True:
                if flag[start]:
                    # Add position to order of permutation
                    permutations[counter] = index[start]
                    flag[start] = False

                    counter += 1
                    break

                if start == size - 1:
                    start = 0
                else:
                    start += 1

        return permutations

    def mating_selection(self, index: int):
        """ Selects `mating_pool_size` distinct parents, either from the neighbourhood or the population based on the
        neighbourhood selection probability.
        """
        parents = list()
        from_neighbourhood = False

        if random.random() <= self.neighbourhood_selection_probability:
            from_neighbourhood = True

        neighbors = self.neighbourhood.get_neighbors(index, self.population)
        parents.append(self.population[index])

        while len(parents) < self.mating_pool_size:
            if from_neighbourhood:
                selected_parent = neighbors[random.randint(0, len(neighbors) - 1)]
            else:
                selected_parent = self.population[random.randint(0, self.population_size - 1)]

            flag = True

            # Check if parent has already been selected
            for parent in parents:
                if parent == selected_parent:
                    flag = False
                    break
            if flag:
                parents.append(selected_parent)

        return parents

    def update_individual(self, index: int, individual: S):
        """ Select pool for replacement.
        """
        size = len(self.neighbourhood.neighborhood[index])
        permutations = self.random_permutations(size)

        c = 0

        for i in range(size):
            k = self.neighbourhood.neighborhood[index][permutations[i]]

            f1 = self.fitness_function.compute(self.population[k].objectives, self.neighbourhood.weight_vectors[k])
            f2 = self.fitness_function.compute(individual.objectives, self.neighbourhood.weight_vectors[k])

            if f2 < f1:
                self.population[k] = individual
                c += 1

            if c >= self.max_number_of_replaced_solutions:
                return

    def init_progress(self) -> None:
        self.population = [self.pop_generator.new(self.problem) for _ in range(self.population_size)]
        self.population = self.evaluate(self.population)

        for individual in self.population:
            self.fitness_function.update(individual.objectives)

    def step(self) -> None:
        permutation = self.random_permutations(self.population_size)

        for i in range(self.population_size):
            index = permutation[i]

            mating_population = self.mating_selection(index)

            self.crossover_operator.current_individual = self.population[index]
            offspring_population = self.reproduction(mating_population)
            offspring_population = self.evaluate(offspring_population)

            self.fitness_function.update(offspring_population[0].objectives)
            self.update_individual(index, offspring_population[0])

    def update_progress(self) -> None:
        observable_data = self.get_observable_data()
        observable_data['SOLUTIONS'] = self.population
        self.observable.notify_all(**observable_data)

    def get_result(self) -> R:
        return self.population

    def get_name(self) -> str:
        return 'MOEAD'
