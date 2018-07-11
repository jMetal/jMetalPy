from jmetal.algorithm import SMPSORP
from jmetal.component.archive import CrowdingDistanceArchiveWithReferencePoint
from jmetal.component.observer import ProgressBarObserver
from jmetal.problem import ZDT1
from jmetal.operator import Polynomial


if __name__ == '__main__':
    problem = ZDT1()
    swarm_size = 100

    reference_points = [[0.0, 0.0]]
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

    progress_bar = ProgressBarObserver(step=swarm_size, maximum=25000)
    algorithm.observable.register(progress_bar)

    algorithm.run()

    print('Algorithm (continuous problem): ' + algorithm.get_name())
    print('Problem: ' + problem.get_name())
    print('Computing time: ' + str(algorithm.total_computing_time))
