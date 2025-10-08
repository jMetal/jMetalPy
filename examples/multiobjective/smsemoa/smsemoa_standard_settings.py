from jmetal.algorithm.multiobjective.smsemoa import SMSEMOA
from jmetal.operator.crossover import SBXCrossover
from jmetal.operator.mutation import PolynomialMutation
from jmetal.problem import DTLZ2
from jmetal.problem.multiobjective.dtlz import DTLZ1, DTLZ3
from jmetal.util.solution import (
    get_non_dominated_solutions,
    print_function_values_to_file,
    print_variables_to_file, read_solutions,
)
from jmetal.util.termination_criterion import StoppingByEvaluations

"""  
 Programa para configurar y ejecutar el algoritmo SMS-EMOA con parámetros estándar.
"""
if __name__ == "__main__":
    problem = DTLZ7()

    max_evaluations = 40000
    algorithm = SMSEMOA(
        problem=problem,
        population_size=50,
        mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables(), distribution_index=20),
        crossover=SBXCrossover(probability=1.0, distribution_index=20),
        termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
    )

    algorithm.run()

    front = get_non_dominated_solutions(algorithm.result())

    # Guardar resultados en fichero
    print_function_values_to_file(front, "FUN." + algorithm.label)
    print_variables_to_file(front, "VAR." + algorithm.label)

    print(f"Algorithm: {algorithm.get_name()}")
    print(f"Problem: {problem.name()}")
    print(f"Computing time: {algorithm.total_computing_time}")
