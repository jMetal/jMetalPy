from jmetal.algorithm.multiobjective.smpso import SMPSO
from jmetal.operator import PolynomialMutation
from jmetal.problem import DTLZ1
from jmetal.util.archive import CrowdingDistanceArchive
from jmetal.util.solution_list.helper import get_numpy_array_from_objectives
from jmetal.util.visualization import InteractivePlot, Plot
from jmetal.util.observer import ProgressBarObserver, VisualizerObserver
from jmetal.util.solution_list import print_function_values_to_file, print_variables_to_file, read_solutions
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.util.visualization.chord_plot import depict_chord_diagram

if __name__ == '__main__':
    problem = DTLZ1(number_of_objectives=3)
    problem.reference_front = read_solutions(file_path='../../resources/reference_front/DTLZ1.3D.pf')

    algorithm = SMPSO(
        problem=problem,
        swarm_size=100,
        mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20),
        leaders=CrowdingDistanceArchive(100),
        termination_criterion=StoppingByEvaluations(max=25000)
    )

    algorithm.observable.register(observer=ProgressBarObserver(max=25000))

    algorithm.run()
    front = algorithm.get_result()

    # Plot front
    plot_tile = algorithm.get_name() + "-" + problem.get_name() + "-" + str(problem.number_of_objectives)

    plot_front = Plot(plot_title='SMPSO-' + problem.get_name(), reference_front=problem.reference_front)
    plot_front.plot([algorithm.get_result(),algorithm.get_result()], filename=plot_tile + '.eps')

    # Plot interactive front
    plot_front = InteractivePlot(plot_title=plot_tile, reference_front=problem.reference_front, axis_labels=problem.obj_labels)
    plot_front.plot(front, filename=plot_tile)

    plot_front = InteractivePlot(plot_title=plot_tile + '-norm', reference_front=problem.reference_front, axis_labels=problem.obj_labels)
    plot_front.plot(front, normalize=True, filename=plot_tile + '-norm')

    # Save variables to file
    print_function_values_to_file(front, 'FUN-' + plot_tile)
    print_variables_to_file(front, 'VAR-' + plot_tile)

    print('Algorithm (continuous problem): ' + algorithm.get_name())
    print('Problem: ' + problem.get_name())
    print('Computing time: ' + str(algorithm.total_computing_time))




