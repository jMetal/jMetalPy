import logging
from typing import List

from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.component.observer import VisualizerObserver
from jmetal.core.solution import FloatSolution
from jmetal.operator.crossover import SBX
from jmetal.operator.mutation import Polynomial
from jmetal.operator.selection import BinaryTournament2Selection
from jmetal.problem.multiobjective.zdt import ZDT1
from jmetal.util.comparator import SolutionAttributeComparator
from jmetal.util.solution_list_output import PrintSolutionList, GraphicSolutionList


def main():
    problem = ZDT1()
    algorithm = NSGAII[FloatSolution, List[FloatSolution]](
        problem=problem,
        population_size=100,
        max_evaluations=25000,
        mutation=Polynomial(1.0 / problem.number_of_variables, distribution_index=20),
        crossover=SBX(1.0, distribution_index=20),
        selection=BinaryTournament2Selection([SolutionAttributeComparator("dominance_ranking"),
                                              SolutionAttributeComparator("crowding_distance", lowest_is_best=False)]))

    observer = VisualizerObserver(problem.get_reference_front())
    dobserver = VisualizerObserver()
    algorithm.observable.register(observer=observer)

    algorithm.run()
    result = algorithm.get_result()

    GraphicSolutionList(title="NSGAII").plot_frontier_to_file(result, 'output')
    GraphicSolutionList(title="NSGAII", reference=problem.get_reference_front()).plot_frontier_to_screen(result)

    #SolutionList[FloatSolution].print_function_values_to_file(result, "NSGAII." + problem.get_name())

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