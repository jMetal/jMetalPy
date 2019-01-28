from dask.distributed import Client

from examples.multiobjective.parallel.zdt1_modified import ZDT1Modified
from jmetal.algorithm.multiobjective.nsgaii import DistributedNSGAII
from jmetal.operator import PolynomialMutation, SBXCrossover, BinaryTournamentSelection
from jmetal.util.comparator import RankingAndCrowdingDistanceComparator

if __name__ == '__main__':
    problem = ZDT1Modified()

    client = Client()

    algorithm = DistributedNSGAII(
        problem=problem,
        population_size=10,
        max_evaluations=100,
        mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20),
        crossover=SBXCrossover(probability=1.0, distribution_index=20),
        selection=BinaryTournamentSelection(comparator=RankingAndCrowdingDistanceComparator()),
        number_of_cores=8,
        client=client
    )

    algorithm.run()
    front = algorithm.get_result()

    print('Algorithm: ' + algorithm.get_name())
    print('Problem: ' + problem.get_name())
    print('Computing time: ' + str(algorithm.total_computing_time))


