import logging

from jmetal.algorithm.multiobjective.smpso import SMPSO
from jmetal.component.archive import CrowdingDistanceArchive
from jmetal.core.solution import FloatSolution
from jmetal.operator.mutation import Polynomial
from jmetal.problem.multiobjective.unconstrained import Kursawe
from jmetal.util.solution_list_output import SolutionListOutput

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    problem = Kursawe()
    algorithm = SMPSO(
        problem=problem,
        swarm_size=100,
        max_evaluations=25000,
        mutation=Polynomial(1.0/problem.number_of_variables, distribution_index=20),
        leaders=CrowdingDistanceArchive(100))

    algorithm.run()
    result = algorithm.get_result()

    SolutionListOutput[FloatSolution].print_function_values_to_file("FUN."+problem.get_name(), result)

    logger.info("Algorithm (continuous problem): " + algorithm.get_name())
    logger.info("Problem: " + problem.get_name())

if __name__ == '__main__':
    main()
