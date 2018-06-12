import logging

from jmetal.algorithm.multiobjective.smpso import SMPSORP
from jmetal.component.archive import CrowdingDistanceArchiveWithReferencePoint
from jmetal.component.observer import VisualizerObserver
from jmetal.operator.mutation import Polynomial
from jmetal.problem.multiobjective.zdt import ZDT1


def main() -> None:
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

    observer = VisualizerObserver()
    algorithm.observable.register(observer=observer)

    algorithm.run()
    result = algorithm.get_result()

    print("Algorithm (continuous problem): " + algorithm.get_name())
    print("Problem: " + problem.get_name())


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
        handlers=[
            logging.FileHandler('jmetalpy.log'),
            logging.StreamHandler()
        ]
    )

    main()