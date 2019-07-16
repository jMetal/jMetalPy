from dask.distributed import Client
from distributed import LocalCluster
from examples.multiobjective.parallel.zdt1_modified import ZDT1Modified
from jmetal.algorithm.multiobjective.nsgaii import DistributedNSGAII
from jmetal.operator import PolynomialMutation, SBXCrossover, BinaryTournamentSelection
from jmetal.util.comparator import RankingAndCrowdingDistanceComparator
from jmetal.util.termination_criterion import StoppingByEvaluations

if __name__ == '__main__':
    problem = ZDT1Modified()

    client = Client(LocalCluster(n_workers=24))

    algorithm = DistributedNSGAII(
        problem=problem,
        population_size=10,
        termination_criterion=StoppingByEvaluations(max=100),
        client=client
    )

    algorithm.run()
    front = algorithm.get_result()

    print('Algorithm: ' + algorithm.get_name())
    print('Problem: ' + problem.get_name())
    print('Computing time: ' + str(algorithm.total_computing_time))

