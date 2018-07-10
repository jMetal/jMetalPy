from jmetal.algorithm import NSGAII
from jmetal.component.observer import VisualizerObserver, ProgressBarObserver
from jmetal.component.comparator import RankingAndCrowdingDistanceComparator
from jmetal.problem import DTLZ1
from jmetal.operator import SBX, Polynomial, BinaryTournamentSelection
from jmetal.util.graphic import ScatterMatplotlib
from jmetal.util.solution_list_output import SolutionList


if __name__ == '__main__':
    problem = DTLZ1()

    algorithm = NSGAII(
        problem=problem,
        population_size=100,
        max_evaluations=25000,
        mutation=Polynomial(probability=1.0 / problem.number_of_variables, distribution_index=20),
        crossover=SBX(probability=1.0, distribution_index=20),
        selection=BinaryTournamentSelection(comparator=RankingAndCrowdingDistanceComparator())
    )

    visualizer = VisualizerObserver(problem)
    progress_bar = ProgressBarObserver(step=100, max=25000)
    algorithm.observable.register(observer=visualizer)
    algorithm.observable.register(observer=progress_bar)

    algorithm.run()
    front = algorithm.get_result()

    # Plot frontier to file
    pareto_front = ScatterMatplotlib(plot_title='NSGAII for DTLZ1', number_of_objectives=problem.number_of_objectives)
    pareto_front.plot(front, reference=problem.get_reference_front(), output='NSGAII-DTLZ1', show=False)

    # Save variables to file
    SolutionList.print_function_values_to_file(front, 'NSGAII.' + problem.get_name())

    print('Algorithm (continuous problem): ' + algorithm.get_name())
    print('Problem: ' + problem.get_name())
    print('Computing time: ' + str(algorithm.total_computing_time))