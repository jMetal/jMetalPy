from typing import TypeVar, List

from jmetal.algorithm.singleobjective.genetic_algorithm import GeneticAlgorithm
from jmetal.config import store
from jmetal.core.operator import Mutation, Crossover
from jmetal.core.problem import Problem
from jmetal.core.solution import Solution
from jmetal.operator import BinaryTournamentSelection
from jmetal.operator.selection import RankingAndFitnessSelection
from jmetal.util.comparator import Comparator
from jmetal.util.comparator import SolutionAttributeComparator
from jmetal.util.evaluator import Evaluator
from jmetal.util.generator import Generator
from jmetal.util.termination_criterion import TerminationCriterion

S = TypeVar('S')
R = TypeVar('R')


class HYPE(GeneticAlgorithm[S, R]):

    def __init__(self,
                 problem: Problem,
                 reference_point: Solution,
                 population_size: int,
                 offspring_population_size: int,
                 mutation: Mutation,
                 crossover: Crossover,
                 termination_criterion: TerminationCriterion = store.default_termination_criteria,
                 population_generator: Generator = store.default_generator,
                 population_evaluator: Evaluator = store.default_evaluator,
                 dominance_comparator: Comparator = store.default_comparator):
        """ This is an implementation of the Hypervolume Estimation Algorithm for Multi-objective Optimization
        proposed in:

        * J. Bader and E. Zitzler. HypE: An Algorithm for Fast Hypervolume-Based Many-Objective
        Optimization. TIK Report 286, Computer Engineering and Networks Laboratory (TIK), ETH
        Zurich, November 2008.

        It uses the Exact Hypervolume-based indicator formulation, which once computed, guides both
        the environmental selection and the binary tournament selection operator

        Please note that as per the publication above, the evaluator and replacement should not be changed
        anyhow. It also requires that Problem() has a reference_point with objective values defined, e.g.

        problem = ZDT1()
        reference_point = FloatSolution(problem.number_of_variables,problem.number_of_objectives, [0], [1])
        reference_point.objectives = [1., 1.]
        """

        selection = BinaryTournamentSelection(
            comparator=SolutionAttributeComparator(key='fitness', lowest_is_best=False))
        self.ranking_fitness = RankingAndFitnessSelection(population_size,
                                                          dominance_comparator=dominance_comparator,
                                                          reference_point=reference_point)
        self.reference_point = reference_point
        self.dominance_comparator = dominance_comparator

        super(HYPE, self).__init__(
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

    def evaluate(self, population: List[S]):
        population = self.population_evaluator.evaluate(population, self.problem)
        population = self.ranking_fitness.compute_hypervol_fitness_values(population, self.reference_point,
                                                                          len(population))
        return population

    def replacement(self, population: List[S], offspring_population: List[S]) -> List[List[S]]:
        join_population = population + offspring_population
        return self.ranking_fitness.execute(join_population)

    def get_result(self) -> R:
        return self.solutions

    def get_name(self) -> str:
        return 'HYPE'
