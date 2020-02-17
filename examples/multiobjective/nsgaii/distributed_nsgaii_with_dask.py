from dask.distributed import Client
from distributed import LocalCluster

from jmetal.algorithm.multiobjective.nsgaii import DistributedNSGAII
from jmetal.operator import PolynomialMutation, SBXCrossover
from jmetal.problem.multiobjective.zdt import ZDT1Modified
from jmetal.util.termination_criterion import StoppingByEvaluations

""" 
Distributed (asynchronous) version of NSGA-II using Dask.
"""

if __name__ == '__main__':
    problem = ZDT1Modified()

    # setup Dask client
    client = Client(LocalCluster(n_workers=24))

    ncores = sum(client.ncores().values())
    print(f'{ncores} cores available')

    # creates the algorithm
    max_evaluations = 25000

    algorithm = DistributedNSGAII(
        problem=problem,
        population_size=100,
        mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20),
        crossover=SBXCrossover(probability=1.0, distribution_index=20),
        termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        number_of_cores=ncores,
        client=client
    )

    algorithm.run()
    front = algorithm.get_result()

    print('Algorithm: ' + algorithm.get_name())
    print('Problem: ' + problem.get_name())
    print('Computing time: ' + str(algorithm.total_computing_time))

