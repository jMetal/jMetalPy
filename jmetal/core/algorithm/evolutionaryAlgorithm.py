from typing import TypeVar, Generic, List

from jmetal.core.solution.binarySolution import BinarySolution

S = TypeVar('S')
R = TypeVar('R')

""" Class representing evolutionary algorithms """
__author__ = "Antonio J. Nebro"


class EvolutionaryAlgorithm(Generic[S, R]):
    def __init__(self):
        self.population = []
        print("EA constructor")

    def create_initial_population(self) -> List[S]:
        pass

    def evaluate_population(self, population: List[S]) -> List[S]:
        pass

    def init_progress(self) -> None:
        pass

    def is_stopping_condition_reached(self) -> bool:
        pass

    def selection(self, population: List[S]) -> List[S]:
        pass

    def reproduction(self, population: List[S]) -> List[S]:
        pass

    def replacement(self, population: List[S], offspring_population: List[S]) -> List[S]:
        pass

    def update_progress(self):
        pass

    def get_result(self)->R:
        pass

    def run(self):
        self.population = self.create_initial_population()
        self.population = self.evaluate_population(self.population)
        self.init_progress()
        while not self.is_stopping_condition_reached():
            print()
            print("------ Begin iteration -------")
            mating_population = self.selection(self.population)
            offspring_population = self.reproduction(mating_population)
            offspring_population = self.evaluate_population(offspring_population)
            self.population = self.replacement(self.population, offspring_population)
            self.update_progress()


class subEA(EvolutionaryAlgorithm[BinarySolution, List[BinarySolution]]):
    def __init__(self):
        super(subEA, self).__init__()
        print("afdsadfasdfasd")
        self.population.append(BinarySolution(2,2))
        self.population.append(BinarySolution(2,2))
        self.population.append(BinarySolution(2,2))
        print("pop size: " + str(len(self.population)))

    def init_progress(self):
        print("init progress in class subEA")

#a = EvolutionaryAlgorithm[BinarySolution, List[BinarySolution]]()
#a.update_progress()
#a.run()


b = subEA()
b.init_progress()
b.update_progress()
