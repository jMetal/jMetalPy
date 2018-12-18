from math import sqrt
from multiprocessing.pool import ThreadPool

import dask
from dask.distributed import Client
from jmetal.component import RankingAndCrowdingDistanceComparator, ProgressBarObserver

from jmetal.algorithm.multiobjective.nsgaii import DistributedNSGAII, DynamicNSGAII
from jmetal.component.observable import TimeCounter
from jmetal.core.problem import FloatProblem, DynamicProblem
from jmetal.core.solution import FloatSolution
from jmetal.operator import Polynomial, SBX, BinaryTournamentSelection
from jmetal.problem.multiobjective.fda import FDA2
from jmetal.util.termination_criterion import StoppingByEvaluations

if __name__ == '__main__':
    problem: DynamicProblem = FDA2()
    time_counter = TimeCounter(problem.observable, 1)
    time_counter.start()

    max_evaluations = 25000
    algorithm = DynamicNSGAII(
        problem=problem,
        population_size=100,
        offspring_population_size=100,
        mating_pool_size=100,
        mutation=Polynomial(probability=1.0 / problem.number_of_variables, distribution_index=20),
        crossover=SBX(probability=1.0, distribution_index=20),
        selection=BinaryTournamentSelection(comparator=RankingAndCrowdingDistanceComparator()),
        termination_criterion=StoppingByEvaluations(max=max_evaluations)
    )

    algorithm.run()
    front = algorithm.get_result()

    print('Algorithm: ' + algorithm.get_name())
    print('Problem: ' + problem.get_name())
    print('Computing time: ' + str(algorithm.total_computing_time))
