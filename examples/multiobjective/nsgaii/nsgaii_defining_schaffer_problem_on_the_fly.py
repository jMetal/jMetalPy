from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.core.problem import OnTheFlyFloatProblem
from jmetal.operator import PolynomialMutation, SBXCrossover
from jmetal.util.solution import (
    get_non_dominated_solutions,
    print_function_values_to_file,
    print_variables_to_file,
)
from jmetal.util.termination_criterion import StoppingByEvaluations

"""  
Program to  configure and run the NSGA-II algorithm configured with standard settings.
"""

if __name__ == "__main__":
    # Defining problem Schaffer on the fly
    def f1(x: [float]):
        return x[0] * x[0]

    def f2(x: [float]):
        return (x[0] - 2) * (x[0] - 2)

    problem = OnTheFlyFloatProblem()
    problem.set_name("Schaffer").add_variable(-1000.0, 1000.0).add_function(f1).add_function(f2)

    max_evaluations = 25000
    algorithm = NSGAII(
        problem=problem,
        population_size=100,
        offspring_population_size=100,
        mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables(), distribution_index=20),
        crossover=SBXCrossover(probability=1.0, distribution_index=20),
        termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
    )

    algorithm.run()
    front = get_non_dominated_solutions(algorithm.result())

    # Save results to file
    print_function_values_to_file(front, "FUN." + algorithm.label)
    print_variables_to_file(front, "VAR." + algorithm.label)

    print(f"Algorithm: {algorithm.get_name()}")
    print(f"Problem: {problem.name()}")
    print(f"Computing time: {algorithm.total_computing_time}")
