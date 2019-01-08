from examples.multiobjective.distributed_nsgaII import ZDT1Modified
from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.component import RankingAndCrowdingDistanceComparator, ProgressBarObserver
from jmetal.component.evaluator import SparkEvaluator
from jmetal.operator import SBXCrossover, Polynomial, BinaryTournamentSelection
from jmetal.util.termination_criterion import StoppingByEvaluations

if __name__ == '__main__':
    problem = ZDT1Modified()

    max_evaluations = 100

    algorithm = NSGAII(
        population_evaluator=SparkEvaluator(),
        problem=problem,
        population_size=10,
        offspring_population_size=10,
        mutation=Polynomial(probability=1.0 / problem.number_of_variables, distribution_index=20),
        crossover=SBXCrossover(probability=1.0, distribution_index=20),
        selection=BinaryTournamentSelection(comparator=RankingAndCrowdingDistanceComparator()),
        termination_criterion=StoppingByEvaluations(max=max_evaluations)
    )
    progress_bar = ProgressBarObserver(max=max_evaluations)
    algorithm.observable.register(observer=progress_bar)

    algorithm.run()
    front = algorithm.get_result()

    print('Algorithm (continuous problem): ' + algorithm.get_name())
    print('Problem: ' + problem.get_name())
    print('Computing time: ' + str(algorithm.total_computing_time))
