from dask.distributed import Client
from distributed import LocalCluster

from jmetal.algorithm.multiobjective.nsgaii import DistributedNSGAII
from jmetal.operator.crossover import SBXCrossover
from jmetal.operator.mutation import PolynomialMutation
from jmetal.problem.multiobjective.zdt import ZDT1Modified
from jmetal.util.termination_criterion import StoppingByEvaluations

""" 
Distributed (asynchronous) version of NSGA-II using Dask.
"""

if __name__ == "__main__":
    problem = ZDT1Modified()

    # setup Dask client
    client = Client(LocalCluster(n_workers=10))

    ncores = sum(client.ncores().values())
    print(f"{ncores} cores available")

    # creates the algorithm
    max_evaluations = 25000

    algorithm = DistributedNSGAII(
        problem=problem,
        population_size=100,
        mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables(), distribution_index=20),
        crossover=SBXCrossover(probability=1.0, distribution_index=20),
        termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        number_of_cores=ncores,
        client=client,
    )

    algorithm.run()
    front = algorithm.result()

    print(f"Algorithm: {algorithm.get_name()}")
    print(f"Problem: {problem.name()}")
    print(f"Computing time: {algorithm.total_computing_time}")
