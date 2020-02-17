from jmetal.lab.visualization import Plot, InteractivePlot
from jmetal.util.observer import ProgressBarObserver, VisualizerObserver
from jmetal.util.solution import print_function_values_to_file, print_variables_to_file, read_solutions

from jmetal.util.termination_criterion import StoppingByEvaluations

from jmetal.algorithm.multiobjective.smpso import SMPSORP
from jmetal.operator import PolynomialMutation
from jmetal.problem import ZDT4, ZDT1
from jmetal.util.archive import CrowdingDistanceArchiveWithReferencePoint


if __name__ == '__main__':
    problem = ZDT1()
    problem.reference_front = read_solutions(filename='resources/reference_front/ZDT1.pf')

    swarm_size = 100

    reference_point = [[0.1, 0.8],[0.6, 0.1]]
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
        termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations)
    )

    algorithm.observable.register(
        observer=VisualizerObserver(reference_front=problem.reference_front, reference_point=reference_point))

    algorithm.run()
    front = algorithm.get_result()

    # Plot front
    plot_front = Plot(title='Pareto front approximation. Problem: ' + problem.get_name(),
                      reference_front=problem.reference_front, axis_labels=problem.obj_labels)
    plot_front.plot(front, label=algorithm.label, filename=algorithm.get_name())

    # Plot interactive front
    plot_front = InteractivePlot(title='Pareto front approximation. Problem: ' + problem.get_name(),
                                 reference_front=problem.reference_front, axis_labels=problem.obj_labels)
    plot_front.plot(front, label=algorithm.label, filename=algorithm.get_name())

    # Save results to file
    print_function_values_to_file(front, 'FUN.' + algorithm.label)
    print_variables_to_file(front, 'VAR.' + algorithm.label)

    print('Algorithm (continuous problem): ' + algorithm.get_name())
    print('Problem: ' + problem.get_name())
    print('Computing time: ' + str(algorithm.total_computing_time))
