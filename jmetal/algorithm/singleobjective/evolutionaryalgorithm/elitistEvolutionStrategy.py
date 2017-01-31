from typing import TypeVar, Generic, List
from copy import deepcopy

from jmetal.core.operator.mutationoperator import MutationOperator
from jmetal.core.algorithm.evolutionaryAlgorithm import EvolutionaryAlgorithm
from jmetal.core.problem.problem import Problem
from jmetal.core.solution.binarySolution import BinarySolution
from jmetal.core.solution.solution import Solution
from jmetal.operator.mutation.bitflip import BitFlip
from jmetal.problem.singleobjective.onemax import OneMax

""" Class representing elitist evolution strategy algorithms """
__author__ = "Antonio J. Nebro"

S = TypeVar('S')
R = TypeVar('R')


class ElitistEvolutionStrategy(EvolutionaryAlgorithm[S, R]):
    def __init__(self, problem: Problem[S], mu: int, lambdA: int, max_evaluations: int, mutation_operator: MutationOperator):
        super(ElitistEvolutionStrategy, self).__init__()
        print("INIT EA")
        self.problem = problem
        self.mu = mu
        self.lambdA = lambdA
        self.max_evaluations = max_evaluations
        self.mutation_operator = mutation_operator
        self.evaluations = 0
        print("MU: " + str(self.mu))
        print("Problem: " + self.problem.get_name())
        print("Max Evals: " + str(self.max_evaluations))

    def init_progress(self):
        print("Init progress in ES")
        self.evaluations = self.mu

    def update_progress(self):
        print("UPDATE progress in ES. Evaluations: " + str(self.evaluations))
        self.evaluations += self.lambdA

    def is_stopping_condition_reached(self) -> bool:
        return self.evaluations >= self.max_evaluations

    def create_initial_population(self) -> List[S]:
        population = []
        for i in range(self.mu):
            population.append(self.problem.create_solution())
        return population

    def evaluate_population(self, population: List[S]):
        print("EVALUATE POPULATION: " + str(len(population)))
        for solution in population:
            self.problem.evaluate(solution)
        return population

    def selection(self, population: List[S]):
        print("SELECTION: " + str(len(population)))
        return population

    def reproduction(self, population: List[S]):
        print("REPRODUCTION: " + str(len(population)))
        offspring_population = []
        for solution in population:
            for j in range((int)(self.lambdA/self.mu)):
                new_solution = deepcopy(solution)
                offspring_population.append(self.mutation_operator.execute(new_solution))

        return offspring_population

    def replacement(self, population: List[S], offspring_population: List[S])\
            -> List[S]:
        print("REPLACEMENT: " + str(len(population)))

        for solution in offspring_population:
            self.population.append(solution)

        population.sort(key=lambda s: s.objectives[0], reverse=True)

        new_population = []
        for i in range(self.mu):
            new_population.append(population[i])

        return new_population

    def get_result(self) -> R:
        print("get result called in ES")
        return self.population[0]



algorithm = ElitistEvolutionStrategy[BinarySolution, BinarySolution]\
    (OneMax(50), mu=1, lambdA=1, max_evaluations= 500, mutation_operator=BitFlip(1.0/256))

algorithm.run()
result = algorithm.get_result()
print("Solution: " + str(result.variables[0]))
print("Fitness:  " + str(result.objectives[0]))
'''
print()
algorithm.population = algorithm.create_initial_population()
print("Population size after create initial population: " + str(len(algorithm.get_population())))

print()
algorithm.population= algorithm.evaluate_population(algorithm.population)
algorithm.init_progress()
print("Population size: " + str(len(algorithm.population)))

print()
mating_population = algorithm.selection(algorithm.population)
print("Mating population size: " + str(len(mating_population)))
'''
#print("pop size: " + str(len(p)))

#algorithm.init_progress()
#algorithm.update_progress()

#result = algorithm.get_result()
#print(len(result))

'''
print()
print("asdvasdfasdfasdfsadfasdfasdfasdfasdfasfa")

class subEA(ElitistEvolutionStrategy[BinarySolution, List[BinarySolution]]):
    def __init__(self, evals: int):
        #super(subEA, self).__init__()
        self.evaluations = evals
        self.lambdA = 525
        print("afdsadfasdfasd")

    def init_progress(self):
        print("init progress in class subEA")

#a = EvolutionaryAlgorithm[BinarySolution, List[BinarySolution]]()
#a.update_progress()
#a.run()


b = subEA(25)
b.init_progress()
b.update_progress()
'''

'''
p = OneMax(1100)
print(p.get_name())
print(p.number_of_bits)
s = p.create_solution()
print(s.get_total_number_of_bits())
'''
