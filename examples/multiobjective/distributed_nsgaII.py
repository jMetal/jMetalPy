from math import sqrt
from multiprocessing.pool import ThreadPool

import dask
from dask.distributed import Client
from jmetal.component import RankingAndCrowdingDistanceComparator, ProgressBarObserver

from jmetal.algorithm.multiobjective.nsgaii import DistributedNSGAII
from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution
from jmetal.operator import Polynomial, SBX, BinaryTournamentSelection
from jmetal.util.solution_list import print_function_values_to_file


class ZDT1Modified(FloatProblem):
    """ Problem ZDT1.

    .. note:: Version including a loop for increasing the computing time of the evaluation functions.
    """

    def __init__(self, number_of_variables: int=30):
        """ :param number_of_variables: Number of decision variables of the problem.
        :param rf_path: Path to the reference front file (if any). Default to None.
        """
        super(ZDT1Modified, self).__init__()
        self.number_of_variables = number_of_variables
        self.number_of_objectives = 2
        self.number_of_constraints = 0

        self.obj_directions = [self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ['f(x)', 'f(y)']

        self.lower_bound = self.number_of_variables * [0.0]
        self.upper_bound = self.number_of_variables * [1.0]

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        g = self.__eval_g(solution)
        h = self.__eval_h(solution.variables[0], g)

        solution.objectives[0] = solution.variables[0]
        solution.objectives[1] = h * g

        s: float = 0.0
        for i in range(10000000):
            s += i * 0.235 / 1.234

        return solution

    def __eval_g(self, solution: FloatSolution):
        g = sum(solution.variables) - solution.variables[0]

        constant = 9.0 / (solution.number_of_variables - 1)
        g = constant * g
        g = g + 1.0

        return g

    def __eval_h(self, f: float, g: float) -> float:
        return 1.0 - sqrt(f / g)

    def get_name(self):
        return 'ZDT11'


if __name__ == '__main__':
    problem = ZDT1Modified()

    dask.config.set(scheduler='threads', pool=ThreadPool(8))
    client = Client()

    algorithm = DistributedNSGAII(
        problem=problem,
        population_size=10,
        max_evaluations=100,
        mutation=Polynomial(probability=1.0 / problem.number_of_variables, distribution_index=20),
        crossover=SBX(probability=1.0, distribution_index=20),
        selection=BinaryTournamentSelection(comparator=RankingAndCrowdingDistanceComparator()),
        number_of_cores=8,
        client=client
    )

    progress_bar = ProgressBarObserver(max=100)
    algorithm.observable.register(observer=progress_bar)

    algorithm.run()
    front = algorithm.get_result()

    print('Algorithm: ' + algorithm.get_name())
    print('Problem: ' + problem.get_name())
    print_function_values_to_file(front, 'FUN.DNSGAII.ZDT1')
    print('Computing time: ' + str(algorithm.total_computing_time))
