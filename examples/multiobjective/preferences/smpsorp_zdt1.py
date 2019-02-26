from jmetal.algorithm.multiobjective.smpso import SMPSORP
from jmetal.operator import PolynomialMutation
from jmetal.problem import ZDT1
from jmetal.util.archive import CrowdingDistanceArchiveWithReferencePoint
from jmetal.util.observer import ProgressBarObserver, VisualizerObserver
from jmetal.util.solution_list import read_solutions, print_function_values_to_file, print_variables_to_file
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.util.visualization import InteractivePlot, Plot

if __name__ == '__main__':
    problem = ZDT1()
    problem.reference_front = read_solutions(filename='../../../resources/reference_front/{}.pf'.format(problem.get_name()))

    swarm_size = 100

    reference_point = [[0.1, 0.8],[0.8, 0.2]]
    archives_with_reference_points = []

    for point in reference_point:
        archives_with_reference_points.append(
            CrowdingDistanceArchiveWithReferencePoint(int(swarm_size / len(reference_point)), point)
        )

    max_evaluations = 50000
    algorithm = SMPSORP(
        problem=problem,
        swarm_size=swarm_size,
        mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20),
        reference_points=reference_point,
        leaders=archives_with_reference_points,
        termination_criterion=StoppingByEvaluations(max=max_evaluations)
    )

    algorithm.observable.register(observer=VisualizerObserver(reference_front=problem.reference_front, reference_point=reference_point))

    algorithm.run()
    front = algorithm.get_result()

    # Plot front
    plot_front = Plot(plot_title='SMPSORP-ZDT1', reference_front=problem.reference_front, reference_point=algorithm.reference_points, axis_labels=problem.obj_labels)
    plot_front.plot(algorithm.get_result(), filename='SMPSORP-ZDT1')

    # Plot interactive front
    plot_front = InteractivePlot(plot_title='SMPSORP-ZDT1', reference_front=problem.reference_front, reference_point=algorithm.reference_points, axis_labels=problem.obj_labels)
    plot_front.plot(front, filename='SMPSORP-ZDT1')

    # Save results to file
    print_function_values_to_file(front, 'FUN.' + algorithm.get_name() + "." + problem.get_name())
    print_variables_to_file(front, 'VAR.'+ algorithm.get_name() + "." + problem.get_name())

    print('Algorithm (continuous problem): ' + algorithm.get_name())
    print('Problem: ' + problem.get_name())
    print('Computing time: ' + str(algorithm.total_computing_time))
