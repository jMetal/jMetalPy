from dask.distributed import Client
<<<<<<< HEAD
from distributed import LocalCluster
=======
>>>>>>> 52e0b172f0c6d651ba08b961a90a382f0a4b8e0f

from examples.multiobjective.parallel.zdt1_modified import ZDT1Modified
from jmetal.algorithm.multiobjective.nsgaii import DistributedNSGAII
from jmetal.operator import PolynomialMutation, SBXCrossover, BinaryTournamentSelection
from jmetal.util.comparator import RankingAndCrowdingDistanceComparator
from jmetal.util.termination_criterion import StoppingByEvaluations

if __name__ == '__main__':
    problem = ZDT1Modified()

<<<<<<< HEAD
    client = Client(LocalCluster(n_workers=24))
=======
    client = Client()
>>>>>>> 52e0b172f0c6d651ba08b961a90a382f0a4b8e0f

    algorithm = DistributedNSGAII(
        problem=problem,
        population_size=10,
<<<<<<< HEAD
        termination_criterion=StoppingByEvaluations(max=100),
=======
>>>>>>> 52e0b172f0c6d651ba08b961a90a382f0a4b8e0f
        mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20),
        crossover=SBXCrossover(probability=1.0, distribution_index=20),
        selection=BinaryTournamentSelection(comparator=RankingAndCrowdingDistanceComparator()),
        number_of_cores=8,
<<<<<<< HEAD
        client=client
=======
        client=client,
        termination_criterion=StoppingByEvaluations(max=100)
>>>>>>> 52e0b172f0c6d651ba08b961a90a382f0a4b8e0f
    )

    algorithm.run()
    front = algorithm.get_result()

    print('Algorithm: ' + algorithm.get_name())
    print('Problem: ' + problem.get_name())
    print('Computing time: ' + str(algorithm.total_computing_time))
<<<<<<< HEAD


=======
>>>>>>> 52e0b172f0c6d651ba08b961a90a382f0a4b8e0f
