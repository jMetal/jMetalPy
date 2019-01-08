from jmetal.algorithm.multiobjective.smpso import SMPSORP
from jmetal.component import ProgressBarObserver, CrowdingDistanceArchiveWithReferencePoint, VisualizerObserver
from jmetal.operator import Polynomial
from jmetal.problem import ZDT1
from jmetal.util.graphic import InteractivePlot
from jmetal.util.solution_list import read_solutions, print_function_values_to_file, print_variables_to_file
from jmetal.util.termination_criterion import StoppingByEvaluations


if __name__ == '__main__':
    problem = ZDT1()
    problem.reference_front = read_solutions(file_path='../../resources/reference_front/{}.pf'.format(problem.get_name()))

    swarm_size = 100

    reference_points = [[0.5, 0.5], [0.1, 0.55]]
    archives_with_reference_points = []

    for point in reference_points:
        archives_with_reference_points.append(
            CrowdingDistanceArchiveWithReferencePoint(swarm_size/len(reference_points), point)
        )

    max_evaluations = 25000
    algorithm = SMPSORP(
        problem=problem,
        swarm_size=swarm_size,
        mutation=Polynomial(probability=1.0 / problem.number_of_variables, distribution_index=20),
        reference_points=reference_points,
        leaders=archives_with_reference_points,
        termination_criterion=StoppingByEvaluations(max=max_evaluations)
    )

    algorithm.observable.register(observer=ProgressBarObserver(max=25000))
    algorithm.observable.register(observer=VisualizerObserver(reference_front=problem.reference_front))

    algorithm.run()
    front = algorithm.get_result()

    # Save results to file
    print_function_values_to_file(front, 'FUN.' + algorithm.get_name() + "." + problem.get_name())
    print_variables_to_file(front, 'VAR.'+ algorithm.get_name() + "." + problem.get_name())

    # Plot frontier to file
    pareto_front = InteractivePlot(plot_title='SMPSORP-ZDT1', reference_point=reference_points, reference_front=problem.reference_front, axis_labels=problem.obj_labels)
    pareto_front.plot(front)
    pareto_front.export_to_html(filename='SMPSORP-ZDT1')

    print('Algorithm (continuous problem): ' + algorithm.get_name())
    print('Problem: ' + problem.get_name())
    print('Computing time: ' + str(algorithm.total_computing_time))
