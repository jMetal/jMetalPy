from jmetal.algorithm.multiobjective.moead import MOEAD_DRA
from jmetal.operator.crossover import DifferentialEvolutionCrossover
from jmetal.operator.mutation import PolynomialMutation
from jmetal.problem.multiobjective.uf import UF1, UF9
from jmetal.util.aggregation_function import Tschebycheff
from jmetal.util.solution import (
    print_function_values_to_file,
    print_variables_to_file,
    read_solutions,
)
from jmetal.util.termination_criterion import StoppingByEvaluations

if __name__ == "__main__":
    problem = UF9()
    problem.reference_front = read_solutions(filename="resources/reference_fronts/UF8.pf")

    max_evaluations = 300000

    algorithm = MOEAD_DRA(
        problem=problem,
        population_size=300,
        crossover=DifferentialEvolutionCrossover(CR=1.0, F=0.5),
        mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables(), distribution_index=20),
        aggregation_function=Tschebycheff(dimension=problem.number_of_objectives()),
        neighbor_size=20,
        neighbourhood_selection_probability=0.9,
        max_number_of_replaced_solutions=2,
        weight_files_path="resources/MOEAD_weights",
        termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
    )

    algorithm.run()
    front = algorithm.result()

    # Save results to file
    print_function_values_to_file(front, "FUN." + algorithm.label)
    print_variables_to_file(front, "VAR." + algorithm.label)

    print(f"Algorithm: {algorithm.get_name()}")
    print(f"Problem: {problem.name()}")
    print(f"Computing time: {algorithm.total_computing_time}")
