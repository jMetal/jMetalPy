from jmetal.algorithm.multiobjective.smpso import SMPSO
from jmetal.operator import PolynomialMutation
from jmetal.problem import DTLZ1
from jmetal.util.archive import CrowdingDistanceArchive
from jmetal.util.observer import ProgressBarObserver
from jmetal.util.solution_list import print_function_values_to_file, print_variables_to_file, read_solutions
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.util.visualization import InteractivePlot, Plot

if __name__ == '__main__':
    problem = DTLZ1(number_of_objectives=5)
    #problem.reference_front = read_solutions(filename='../../resources/reference_front/DTLZ1.3D.pf')

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
    label = '{}-{} with {} objectives'.format(algorithm.get_name(), problem.get_name(), problem.number_of_objectives)

    plot_front = Plot(plot_title='Pareto front approximation', reference_front=problem.reference_front, axis_labels=problem.obj_labels)
    plot_front.plot(front, label=label, filename=label)

    # Plot interactive front
    plot_front = InteractivePlot(plot_title='Pareto front approximation', reference_front=problem.reference_front, axis_labels=problem.obj_labels)
    plot_front.plot(front, label=label, filename=label, normalize=True)

    # Save variables to file
    print_function_values_to_file(front, 'FUN-' + label)
    print_variables_to_file(front, 'VAR-' + label)

    print('Algorithm (continuous problem): ' + algorithm.get_name())
    print('Problem: ' + problem.get_name())
    print('Computing time: ' + str(algorithm.total_computing_time))




