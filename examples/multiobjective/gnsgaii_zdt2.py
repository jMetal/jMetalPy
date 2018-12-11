from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.component import ProgressBarObserver, RankingAndCrowdingDistanceComparator
from jmetal.component.comparator import GDominanceComparator
from jmetal.component.observer import VisualizerObserver
from jmetal.operator import SBX, Polynomial, BinaryTournamentSelection
from jmetal.problem import ZDT2
from jmetal.util.graphic import FrontPlot
from jmetal.util.solution_list import print_function_values_to_file, print_variables_to_file, read_front
from jmetal.util.termination_criteria import StoppingByEvaluations

"""
.. module:: GNSGA-II
   :platform: Unix, Windows
   :synopsis: GNSGA-II (Non-dominance Sorting Genetic Algorithm II with preference articulation
   based on a reference point).

.. moduleauthor:: Antonio J. Nebro <antonio@lcc.uma.es>
"""

if __name__ == '__main__':
    problem = ZDT2()
    problem.reference_front = read_front(file_path='../../resources/reference_front/{}.pf'.format(problem.get_name()))

    reference_point = [0.5, 0.5]

    algorithm = NSGAII(
        problem=problem,
        population_size=100,
        offspring_size=100,
        mating_pool_size=100,
        mutation=Polynomial(probability=1.0 / problem.number_of_variables, distribution_index=20),
        crossover=SBX(probability=1.0, distribution_index=20),
        selection=BinaryTournamentSelection(comparator=RankingAndCrowdingDistanceComparator()),
        dominance_comparator=GDominanceComparator(reference_point),
        termination_criteria=StoppingByEvaluations(max=25000),
        # termination_criteria=StoppingByTime(max_seconds=20),
        # termination_criteria=StoppingByQualityIndicator(quality_indicator=HyperVolume([1.0, 1.0]), expected_value=0.5, degree=0.95)
    )

    progress_bar = ProgressBarObserver(max=25000)
    algorithm.observable.register(observer=progress_bar)
    algorithm.observable.register(observer=VisualizerObserver())

    algorithm.run()
    front = algorithm.get_result()

    # Plot frontier to file
    pareto_front = FrontPlot(plot_title='GNSGAII-ZDT2', axis_labels=problem.obj_labels)
    pareto_front.plot(front, reference_front=problem.reference_front)
    pareto_front.to_html(filename='GNSGAII-ZDT2')

    # Save variables to file
    print_function_values_to_file(front, 'FUN.GNSGAII.ZDT2')
    print_variables_to_file(front, 'VAR.GNSGAII.ZDT2')

    print('Algorithm (continuous problem): ' + algorithm.get_name())
    print('Problem: ' + problem.get_name())
    print('Computing time: ' + str(algorithm.total_computing_time))
