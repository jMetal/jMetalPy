import logging

from jmetal.algorithm.multiobjective.smpso import SMPSO
from jmetal.component.archive import CrowdingDistanceArchive
from jmetal.component.observer import VisualizerObserver
from jmetal.core.solution import FloatSolution
from jmetal.operator.mutation import Polynomial
from jmetal.problem.multiobjective.dtlz import DTLZ1
from jmetal.problem.multiobjective.zdt import ZDT1
from jmetal.util.solution_list_output import SolutionListOutput


def main() -> None:
    problem = ZDT1()
    algorithm = SMPSO(
        problem=problem,
        swarm_size=100,
        max_evaluations=25000,
        mutation=Polynomial(probability=1.0/problem.number_of_variables, distribution_index=20),
        leaders=CrowdingDistanceArchive(100)
    )

    observer = VisualizerObserver(replace=True)
    algorithm.observable.register(observer=observer)

    algorithm.run()
    result = algorithm.get_result()

    SolutionListOutput[FloatSolution].plot_frontier_to_screen(result, None, title=problem.get_name())
    SolutionListOutput[FloatSolution].print_function_values_to_file(result, "SMPSO." + problem.get_name())

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