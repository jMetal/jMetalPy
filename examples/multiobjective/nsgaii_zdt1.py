from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.component import RankingAndCrowdingDistanceComparator, VisualizerObserver
from jmetal.operator import SBX, Polynomial, BinaryTournamentSelection
from jmetal.problem import ZDT1
from jmetal.util.solution_list import read_solutions
from jmetal.util.termination_criteria import StoppingByEvaluations

if __name__ == '__main__':
    problem = ZDT1()
    problem.reference_front = read_solutions(file_path='../../resources/reference_front/{}.pf'.format(problem.get_name()))

    algorithm = NSGAII(
        problem=problem,
        population_size=100,
        offspring_size=100,
        mating_pool_size=100,
        mutation=Polynomial(probability=1.0 / problem.number_of_variables, distribution_index=20),
        crossover=SBX(probability=1.0, distribution_index=20),
        selection=BinaryTournamentSelection(comparator=RankingAndCrowdingDistanceComparator()),
        termination_criteria=StoppingByEvaluations(max=25000)
    )

    algorithm.observable.register(observer=VisualizerObserver(reference_front=problem.reference_front))

    algorithm.run()
    front = algorithm.get_result()

    print('Algorithm (continuous problem): ' + algorithm.get_name())
    print('Problem: ' + problem.get_name())
    print('Computing time: ' + str(algorithm.total_computing_time))
