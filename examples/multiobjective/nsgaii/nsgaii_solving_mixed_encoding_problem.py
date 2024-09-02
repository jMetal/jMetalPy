from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.operator.crossover import CompositeCrossover, IntegerSBXCrossover
from jmetal.operator.crossover import SBXCrossover
from jmetal.operator.mutation import CompositeMutation
from jmetal.operator.mutation import IntegerPolynomialMutation, PolynomialMutation
from jmetal.problem.multiobjective.unconstrained import MixedIntegerFloatProblem
from jmetal.util.observer import VisualizerObserver
from jmetal.util.solution import (
    get_non_dominated_solutions,
    print_function_values_to_file,
    print_variables_to_file,
)
from jmetal.util.termination_criterion import StoppingByEvaluations

if __name__ == "__main__":
    problem = MixedIntegerFloatProblem(60, 20, 100, -100, -1000, 1000)

    max_evaluations = 25000
    algorithm = NSGAII(
        problem=problem,
        population_size=100,
        offspring_population_size=100,
        mutation=CompositeMutation([IntegerPolynomialMutation(0.01, 20), PolynomialMutation(0.01, 20.0)]),
        crossover=CompositeCrossover(
            [
                IntegerSBXCrossover(probability=1.0, distribution_index=20),
                SBXCrossover(probability=1.0, distribution_index=20),
            ]
        ),
        termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
    )

    algorithm.observable.register(observer=VisualizerObserver())

    algorithm.run()
    front = get_non_dominated_solutions(algorithm.result())

    # Save results to file
    print_function_values_to_file(front, "FUN." + algorithm.label)
    print_variables_to_file(front, "VAR." + algorithm.label)

    print(f"Algorithm: {algorithm.get_name()}")
    print(f"Problem: {problem.name()}")
    print(f"Computing time: {algorithm.total_computing_time}")
