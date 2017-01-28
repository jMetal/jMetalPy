from typing import TypeVar, Generic, List

Solution = TypeVar('S')
Result = TypeVar('R')

""" Class representing evolutionary algorithms """
__author__ = "Antonio J. Nebro"


class EvolutionaryAlgorithm((Generic[Solution], Result)):
    def __init__(self):
        self.population = List[Solution]

    def create_initial_population(self) -> List[Solution]:
        pass

    def evaluate_population(self, population: List[Solution]) -> List[Solution]:
        pass

    def init_progress(self) -> None:
        pass

    def is_stopping_condition_reached(self) -> bool:
        pass

    def selection(self, population: List[Solution]) -> List[Solution]:
        pass

    def reproduction(self, population: List[Solution]) -> List[Solution]:
        pass

    def replacement(self, population: List[Solution], offspring_population: List[Solution]) -> List[Solution]:
        pass

    def update_progress(self):
        pass

    def get_result(self)->Result:
        pass

    def run(self):
        self.population = self.create_initial_population()
        self.population = self.evaluate_population(self.population)
        self.init_progress()
        while not self.is_stopping_condition_reached():
            mating_population = self.selection(self.population)
            offspring_population = self.reproduction(mating_population)
            offspring_population = self.evaluate_population(offspring_population)
            self.population = self.replacement(self.population, offspring_population)
            self.update_progress()




