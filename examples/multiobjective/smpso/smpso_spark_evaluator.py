
from jmetal.algorithm.multiobjective.smpso import SMPSO
from jmetal.operator.mutation import PolynomialMutation
from jmetal.util.archive import CrowdingDistanceArchive
from jmetal.util.evaluator import SparkEvaluator
from jmetal.util.solution import (
    print_function_values_to_file,
    print_variables_to_file,
    read_solutions,
)
from jmetal.util.termination_criterion import StoppingByEvaluations

from src.jmetal.problem.multiobjective.zdt import ZDT1Modified

if __name__ == "__main__":
    problem = ZDT1Modified()
    problem.reference_front = read_solutions(filename="resources/reference_fronts/ZDT1.pf")

    max_evaluations = 100
    algorithm = SMPSO(
        problem=problem,
        swarm_size=10,
        mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables(), distribution_index=20),
        leaders=CrowdingDistanceArchive(10),
        termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        swarm_evaluator=SparkEvaluator(),
    )

    algorithm.run()
    front = algorithm.result()

    # Save results to file
    print_function_values_to_file(front, "FUN." + algorithm.get_name() + "." + problem.get_name())
    print_variables_to_file(front, "VAR." + algorithm.get_name() + "." + problem.get_name())

    print(f"Algorithm: {algorithm.get_name()}")
    print(f"Problem: {problem.name()}")
    print(f"Computing time: {algorithm.total_computing_time}")
