import logging
from typing import List

from jmetal.component.evaluator import SequentialEvaluator, DaskMultithreadedEvaluator, ParallelEvaluator

from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.component.observer import VisualizerObserver
from jmetal.core.solution import FloatSolution
from jmetal.operator.crossover import SBX
from jmetal.operator.mutation import Polynomial
from jmetal.operator.selection import BinaryTournament2Selection
from jmetal.problem.multiobjective.zdt import ZDT1
from jmetal.util.comparator import SolutionAttributeComparator
from jmetal.util.solution_list_output import SolutionListOutput
from jmetal.problem.multiobjective.unconstrained import Kursawe

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    problem = ZDT1()
    algorithm = NSGAII[FloatSolution, List[FloatSolution]](
        problem=problem,
        population_size=100,
        max_evaluations=25000,
        #evaluator=DaskMultithreadedEvaluator(),
        mutation=Polynomial(1.0/problem.number_of_variables, distribution_index=20),
        crossover=SBX(1.0, distribution_index=20),
        # selection=BinaryTournamentSelection(RankingAndCrowdingDistanceComparator()))
        selection=BinaryTournament2Selection([SolutionAttributeComparator("dominance_ranking"),
                                              SolutionAttributeComparator("crowding_distance", lowest_is_best=False)]))

    observer = VisualizerObserver(animation_speed=1 * 10e-8)
    algorithm.observable.register(observer=observer)

    algorithm.run()
    result = algorithm.get_result()

    SolutionListOutput[FloatSolution].plot_frontier_to_file(result, None, file_name="NSGAII." + problem.get_name(),
                                                            output_format='png', dpi=200)
    SolutionListOutput[FloatSolution].plot_frontier_to_screen(result)
    SolutionListOutput[FloatSolution].print_function_values_to_file("NSGAII." + problem.get_name(), result)

    logger.info("Algorithm (continuous problem): " + algorithm.get_name())
    logger.info("Problem: " + problem.get_name())
    logger.info("Computing time: " + str(algorithm.total_computing_time))


if __name__ == '__main__':
    main()
