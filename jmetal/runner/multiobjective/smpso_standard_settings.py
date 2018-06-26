import logging

from jmetal.util.graphic import ScatterBokeh

from jmetal.algorithm.multiobjective.smpso import SMPSO
from jmetal.component.archive import CrowdingDistanceArchive
from jmetal.component.observer import VisualizerObserver
from jmetal.operator.mutation import Polynomial
from jmetal.problem.multiobjective.zdt import ZDT1
from jmetal.problem.multiobjective.dtlz import DTLZ1


def main() -> None:
    problem = DTLZ1()
    algorithm = SMPSO(
        problem=problem,
        swarm_size=100,
        max_evaluations=25000,
        mutation=Polynomial(probability=1.0/problem.number_of_variables, distribution_index=20),
        leaders=CrowdingDistanceArchive(100)
    )

    observer = VisualizerObserver(problem)
    algorithm.observable.register(observer=observer)

    algorithm.run()
    result = algorithm.get_result()

    # Plot frontier
    pareto_front = ScatterBokeh(plot_title='SMPSO for DTLZ1', number_of_objectives=problem.number_of_objectives)
    pareto_front.plot(result, reference=problem.get_reference_front(), output='output')

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