import logging

from jmetal.algorithm.multiobjective.smpso import SMPSORP
from jmetal.component.archive import CrowdingDistanceArchiveWithReferencePoint
from jmetal.component.observer import AlgorithmObserver
from jmetal.core.solution import FloatSolution
from jmetal.operator.mutation import Polynomial
from jmetal.problem.multiobjective.zdt import ZDT1
from jmetal.util.solution_list_output import SolutionListOutput

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    problem = ZDT1()
    swarm_size = 100

    reference_points = [[0.8, 0.2], [0.4, 0.6]]
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

    observer = AlgorithmObserver(animation_speed=1*10e-8)
    algorithm.observable.register(observer=observer)

    algorithm.run()
    result = algorithm.get_result()

    SolutionListOutput[FloatSolution].plot_scatter_to_screen(result)

    logger.info("Algorithm (continuous problem): " + algorithm.get_name())
    logger.info("Problem: " + problem.get_name())


if __name__ == '__main__':
    main()
