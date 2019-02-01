from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.operator import SBXCrossover, PolynomialMutation, BinaryTournamentSelection
from jmetal.problem import ZDT2
from jmetal.util.comparator import GDominanceComparator, RankingAndCrowdingDistanceComparator
from jmetal.util.observer import ProgressBarObserver, VisualizerObserver
from jmetal.util.solution_list import print_function_values_to_file, print_variables_to_file, read_solutions
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.util.visualization import Plot, InteractivePlot

"""
.. module:: GNSGA-II
   :platform: Unix, Windows
   :synopsis: GNSGA-II (Non-dominance Sorting Genetic Algorithm II with preference articulation
   based on a reference point).

.. moduleauthor:: Antonio J. Nebro <antonio@lcc.uma.es>
"""

if __name__ == '__main__':
    problem = ZDT2()
    problem.reference_front = read_solutions(filename='../../../resources/reference_front/{}.pf'.format(problem.get_name()))

    reference_point = [0.8, 0.5]

    max_evaluations = 25000
    algorithm = NSGAII(
        problem=problem,
        population_size=100,
        offspring_population_size=100,
        mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20),
        crossover=SBXCrossover(probability=1.0, distribution_index=20),
        selection=BinaryTournamentSelection(comparator=RankingAndCrowdingDistanceComparator()),
        dominance_comparator=GDominanceComparator(reference_point),
        termination_criterion=StoppingByEvaluations(max=max_evaluations)
    )

    algorithm.observable.register(observer=ProgressBarObserver(max=max_evaluations))
    algorithm.observable.register(observer=VisualizerObserver(reference_front=problem.reference_front, reference_point=(reference_point)))

    algorithm.run()
    front = algorithm.get_result()

    # Plot front
    plot_front = Plot(plot_title='Pareto front approximation', axis_labels=problem.obj_labels, reference_point=reference_point, reference_front=problem.reference_front)
    plot_front.plot(front, label='gNSGAII-ZDT1', filename='gNSGAII-ZDT1')

    # Plot interactive front
    plot_front = InteractivePlot(plot_title='Pareto front approximation', axis_labels=problem.obj_labels, reference_point=reference_point, reference_front=problem.reference_front)
    plot_front.plot(front, label='gNSGAII-ZDT1', filename='gNSGAII-ZDT1')

    # Save results to file
    print_function_values_to_file(front, 'FUN.' + algorithm.get_name() + "." + problem.get_name())
    print_variables_to_file(front, 'VAR.'+ algorithm.get_name() + "." + problem.get_name())

    print('Algorithm (continuous problem): ' + algorithm.get_name())
    print('Problem: ' + problem.get_name())
    print('Computing time: ' + str(algorithm.total_computing_time))
