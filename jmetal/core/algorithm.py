import logging
import threading
import time
from abc import abstractmethod, ABC
from typing import TypeVar, Generic, List

from jmetal.config import store
from jmetal.core.problem import Problem
from jmetal.core.solution import FloatSolution

LOGGER = logging.getLogger('jmetal')

S = TypeVar('S')
R = TypeVar('R')

"""
.. module:: algorithm
   :platform: Unix, Windows
   :synopsis: Templates for algorithms.

.. moduleauthor:: Antonio J. Nebro <antonio@lcc.uma.es>, Antonio Benítez-Hidalgo <antonio.b@uma.es>
"""


class Algorithm(Generic[S, R], threading.Thread, ABC):

    def __init__(self):
        """
        """
        threading.Thread.__init__(self)

        self.evaluations = 0
        self.start_computing_time = 0
        self.total_computing_time = 0

        self.observable = store.default_observable

    @abstractmethod
    def create_initial_solutions(self) -> List[S]:
        """ Creates the initial list of solutions of a metaheuristic"""
        pass

    @abstractmethod
    def evaluate(self, solution_list:List[S]) -> List[S]:
        """ Evaluates a solution list """
        pass

    @abstractmethod
    def init_progress(self) -> None:
        """ Initialize the algorithm. """
        pass

    @abstractmethod
    def stopping_condition_is_met(self) -> bool:
        """ Stopping condition test"""
        pass

    @abstractmethod
    def step(self) -> None:
        """ Performs one iteration/step of the algorithm's loop. """
        pass

    @abstractmethod
    def update_progress(self) -> None:
        """ Update the progress after each iteration. """
        pass

    def run(self):
        """ Execute the algorithm. """
        self.start_computing_time = time.time()

        solution_list = self.create_initial_solutions()
        solution_list = self.evaluate(solution_list)

        LOGGER.debug('Initializing progress')
        self.init_progress()

        LOGGER.debug('Running main loop until termination criteria is met')
        while self.stopping_condition_is_met():
            self.step(solution_list)
            self.update_progress()

        self.total_computing_time = time.time() - self.start_computing_time

    def get_observable_data(self) -> dict:
        """ Get observable data, with the information that will be send to all observers each time. """
        ctime = time.time() - self.start_computing_time
        return {'PROBLEM': self.problem, 'EVALUATIONS': self.evaluations, 'SOLUTIONS': [], 'COMPUTING_TIME': ctime}

    @abstractmethod
    def get_result(self) -> R:
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass


class DynamicAlgorithm(ABC):
    @abstractmethod
    def restart(self) -> None:
        pass


class EvolutionaryAlgorithm(Algorithm[S, R], ABC):

    def __init__(self,
                 problem: Problem[S],
                 population_size: int,
                 offspring_population_size: int):
        super(EvolutionaryAlgorithm, self).__init__(
            problem=problem
        )
        self.problem = problem
        self.population_size = population_size
        self.offspring_population_size = offspring_population_size

    @abstractmethod
    def selection(self, population: List[S]) -> List[S]:
        """ Select the best-fit individuals for reproduction (parents). """
        pass

    @abstractmethod
    def reproduction(self, population: List[S]) -> List[S]:
        """ Breed new individuals through crossover and mutation operations to give birth to offspring. """
        pass

    @abstractmethod
    def replacement(self, population: List[S], offspring_population: List[S]) -> List[S]:
        """ Replace least-fit population with new individuals. """
        pass

    def init_progress(self) -> None:
        observable_data = self.get_observable_data()
        self.observable.notify_all(**observable_data)

    def step(self) -> None:
        mating_population = self.selection(self.solution_list)
        offspring_population = self.reproduction(mating_population)
        offspring_population = self.evaluate(offspring_population)
        self.solution_list = self.replacement(self.solution_list, offspring_population)

    def update_progress(self) -> None:
        observable_data = self.get_observable_data()
        observable_data['SOLUTIONS'] = self.solution_list
        self.observable.notify_all(**observable_data)


class ParticleSwarmOptimization(Algorithm[FloatSolution, List[FloatSolution]], ABC):

    def __init__(self,
                 problem: Problem,
                 swarm_size: int):
        super(ParticleSwarmOptimization, self).__init__(problem=problem)
        self.swarm_size = swarm_size

    @abstractmethod
    def initialize_velocity(self, swarm: List[FloatSolution]) -> None:
        pass

    @abstractmethod
    def initialize_particle_best(self, swarm: List[FloatSolution]) -> None:
        pass

    @abstractmethod
    def initialize_global_best(self, swarm: List[FloatSolution]) -> None:
        pass

    @abstractmethod
    def update_velocity(self, swarm: List[FloatSolution]) -> None:
        pass

    @abstractmethod
    def update_particle_best(self, swarm: List[FloatSolution]) -> None:
        pass

    @abstractmethod
    def update_global_best(self, swarm: List[FloatSolution]) -> None:
        pass

    @abstractmethod
    def update_position(self, swarm: List[FloatSolution]) -> None:
        pass

    @abstractmethod
    def perturbation(self, swarm: List[FloatSolution]) -> None:
        pass

    def init_progress(self) -> None:
        self.initialize_velocity(self.solution_list)
        self.initialize_particle_best(self.solution_list)
        self.initialize_global_best(self.solution_list)

    def step(self):
        self.update_velocity(self.solution_list)
        self.update_position(self.solution_list)
        self.perturbation(self.solution_list)
        self.solution_list = self.evaluate(self.solution_list)
        self.update_global_best(self.solution_list)
        self.update_particle_best(self.solution_list)

    def update_progress(self) -> None:
        observable_data = self.get_observable_data()
        observable_data['SOLUTIONS'] = self.swarm
        self.observable.notify_all(**observable_data)
