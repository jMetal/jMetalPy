from jmetal.algorithm import NSGAII
from jmetal.component import VisualizerObserver, ProgressBarObserver, RankingAndCrowdingDistanceComparator
from jmetal.problem import ZDT1
from jmetal.operator import SBX, Polynomial, BinaryTournamentSelection
from jmetal.util import ScatterMatplotlib, SolutionList


if __name__ == '__main__':
    problem = ZDT1()

    algorithm = NSGAII(
        problem=problem,
        population_size=100,
        max_evaluations=25000,
        mutation=Polynomial(probability=1.0 / problem.number_of_variables, distribution_index=20),
        crossover=SBX(probability=1.0, distribution_index=20),
        selection=BinaryTournamentSelection(comparator=RankingAndCrowdingDistanceComparator())
    )

    algorithm.run()
    front = algorithm.get_result()

    # Plot frontier to file
    pareto_front = ScatterMatplotlib(plot_title='NSGAII for ZDT1', number_of_objectives=problem.number_of_objectives)
    pareto_front.plot(front, reference=problem.get_reference_front(), output='NSGAII-DTLZ1', show=False)

    # Save variables to file

    print('Algorithm (continuous problem): ' + algorithm.get_name())
    print('Problem: ' + problem.get_name())
    print('Computing time: ' + str(algorithm.total_computing_time))