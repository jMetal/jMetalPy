import logging

from jmetal.component.observer import ProgressBarObserver
from jmetal.problem.multiobjective.zdt import ZDT1
from jmetal.algorithm.multiobjective.smpso import SMPSO
from jmetal.component.archive import CrowdingDistanceArchive
from jmetal.operator.mutation import Polynomial


def main() -> None:
    problem = ZDT1()

    algorithm = SMPSO(
        problem=problem,
        swarm_size=100,
        max_evaluations=25000,
        mutation=Polynomial(probability=1.0/problem.number_of_variables, distribution_index=20),
        leaders=CrowdingDistanceArchive(100)
    )

    progress_bar = ProgressBarObserver(step=100, max=25000)
    algorithm.observable.register(progress_bar)

    algorithm.run()

    print("Algorithm (continuous problem): " + algorithm.get_name())
    print("Problem: " + problem.get_name())
    print("Computing time: " + str(algorithm.total_computing_time))


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