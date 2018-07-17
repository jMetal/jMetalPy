from jmetal.algorithm import SMPSORP
from jmetal.component import ProgressBarObserver, VisualizerObserver, CrowdingDistanceArchiveWithReferencePoint
from jmetal.problem import ZDT1
from jmetal.operator import Polynomial
from jmetal.util import ScatterPlot


def points_to_solutions(points):
    solutions = []
    for i, _ in enumerate(points):
        point = problem.create_solution()
        point.objectives = points[i]
        solutions.append(point)

    return solutions


if __name__ == '__main__':
    problem = ZDT1(rf_path='../../resources/reference_front/ZDT1.pf')

    swarm_size = 100

    reference_points = [[0.5, 0.5], [0.2, 0.8]]
    archives_with_reference_points = []

    for point in reference_points:
        archives_with_reference_points.append(
            CrowdingDistanceArchiveWithReferencePoint(swarm_size, point)
        )

    algorithm = SMPSORP(
        problem=problem,
        swarm_size=swarm_size,
        max_evaluations=25000,
        mutation=Polynomial(probability=1.0/problem.number_of_variables, distribution_index=20),
        reference_points=reference_points,
        leaders=archives_with_reference_points
    )

    observer = VisualizerObserver()
    algorithm.observable.register(observer=observer)

    progress_bar = ProgressBarObserver(step=swarm_size, maximum=25000)
    algorithm.observable.register(observer=progress_bar)

    algorithm.run()
    front = algorithm.get_result()

    # Plot frontier to file
    pareto_front = ScatterPlot(plot_title='SMPSORP-ZDT1', axis_labels=problem.obj_labels)
    pareto_front.plot(front, reference_front=problem.reference_front, show=False)
    pareto_front.add_data(points_to_solutions(reference_points), legend='reference points')
    pareto_front.show()

    print('Algorithm (continuous problem): ' + algorithm.get_name())
    print('Problem: ' + problem.get_name())
    print('Computing time: ' + str(algorithm.total_computing_time))