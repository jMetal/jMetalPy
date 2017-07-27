import logging
from typing import List

from jmetal.problem.multiobjective.constrained import Srinivas, Tanaka
from jmetal.problem.multiobjective.unconstrained import Schaffer, Fonseca, Viennet2

from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.component.observer import AlgorithmObserver
from jmetal.core.solution import FloatSolution
from jmetal.operator.crossover import SBX
from jmetal.operator.mutation import Polynomial
from jmetal.operator.selection import BinaryTournament, BinaryTournament2
from jmetal.problem.multiobjective.zdt import ZDT1, ZDT2, ZDT3, ZDT4, ZDT6
from jmetal.util.comparator import RankingAndCrowdingDistanceComparator, SolutionAttributeComparator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    problem = Tanaka()
    algorithm = NSGAII[FloatSolution, List[FloatSolution]](
        problem,
        population_size=100,
        max_evaluations=25000,
        mutation=Polynomial(1.0/problem.number_of_variables, distribution_index=20),
        crossover=SBX(1.0, distribution_index=20),
        #selection=BinaryTournament(RankingAndCrowdingDistanceComparator()))
        selection = BinaryTournament2([SolutionAttributeComparator("dominance_ranking"),
                                       SolutionAttributeComparator("crowding_distance", lowest_is_best=False)]))

    observer = AlgorithmObserver(animation_speed=1*10e-8)
    algorithm.observable.register(observer=observer)

    algorithm.run()

    logger.info("Algorithm (continuous problem): " + algorithm.get_name())
    logger.info("Problem: " + problem.get_name())

if __name__ == '__main__':
    main()
