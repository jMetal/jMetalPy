from jmetal.algorithm import NSGAII
from jmetal.problem import DTLZ1
from jmetal.operator import SBX, Polynomial, BinaryTournamentSelection
from jmetal.component import ProgressBarObserver, VisualizerObserver, RankingAndCrowdingDistanceComparator
from jmetal.util import FrontPlot, SolutionList


if __name__ == '__main__':
    problem = DTLZ1(rf_path='../../resources/reference_front/DTLZ1.pf')

    algorithm = NSGAII(
        problem=problem,
        population_size=100,
        max_evaluations=50000,
        mutation=Polynomial(probability=1.0 / problem.number_of_variables, distribution_index=20),
        crossover=SBX(probability=1.0, distribution_index=20),
        selection=BinaryTournamentSelection(comparator=RankingAndCrowdingDistanceComparator())
    )

    progress_bar = ProgressBarObserver(step=100, maximum=50000)
    visualizer = VisualizerObserver()
    algorithm.observable.register(observer=progress_bar)
    algorithm.observable.register(observer=visualizer)

    algorithm.run()
    front = algorithm.get_result()

    # Plot frontier to file
    pareto_front = FrontPlot(plot_title='NSGAII-DTLZ1', axis_labels=problem.obj_labels)
    pareto_front.plot(front, reference_front=problem.reference_front)
    pareto_front.to_html(filename='NSGAII-DTLZ1')

    # Save variables to file
    SolutionList.print_function_values_to_file(front, 'FUN.NSGAII.DTLZ1')
    SolutionList.print_variables_to_file(front, 'VAR.NSGAII.DTLZ1')

    print('Algorithm (continuous problem): ' + algorithm.get_name())
    print('Problem: ' + problem.get_name())
    print('Computing time: ' + str(algorithm.total_computing_time))
