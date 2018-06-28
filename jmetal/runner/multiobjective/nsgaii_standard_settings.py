import logging

from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.component.observer import ProgressBarObserver
from jmetal.operator.crossover import SBX
from jmetal.operator.mutation import Polynomial
from jmetal.operator.selection import BinaryTournament2Selection
from jmetal.problem.multiobjective.dtlz import DTLZ1
from jmetal.util.comparator import SolutionAttributeComparator


def main() -> None:
    problem = DTLZ1()

    algorithm = NSGAII(
        problem=problem,
        population_size=100,
        max_evaluations=25000,
        mutation=Polynomial(1.0 / problem.number_of_variables, distribution_index=20),
        crossover=SBX(1.0, distribution_index=20),
        selection=BinaryTournament2Selection([SolutionAttributeComparator('dominance_ranking'),
                                              SolutionAttributeComparator('crowding_distance', lowest_is_best=False)]))

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