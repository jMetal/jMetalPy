import logging

from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.component.observer import VisualizerObserver
from jmetal.core.solution import FloatSolution
from jmetal.operator.crossover import SBX
from jmetal.operator.mutation import Polynomial
from jmetal.operator.selection import BinaryTournament2Selection
from jmetal.problem.multiobjective.dtlz import DTLZ1
from jmetal.component.comparator import SolutionAttributeComparator
from jmetal.util.graphic import ScatterMatplotlib
from jmetal.util.solution_list_output import SolutionList


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

    observer = VisualizerObserver(problem)
    algorithm.observable.register(observer=observer)

    algorithm.run()
    result = algorithm.get_result()

    # Plot frontier
    pareto_front = ScatterMatplotlib(plot_title='NSGAII for DTLZ1', number_of_objectives=problem.number_of_objectives)
    pareto_front.plot(result, reference=problem.get_reference_front(), output='output2')

    # Save variables to file
    SolutionList[FloatSolution].print_function_values_to_file(result, 'NSGAII.' + problem.get_name())

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