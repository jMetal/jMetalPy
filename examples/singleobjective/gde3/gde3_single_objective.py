from jmetal.algorithm.multiobjective.gde3 import GDE3
from jmetal.problem.singleobjective.unconstrained import Rastrigin
from jmetal.util.solution import print_function_values_to_file, print_variables_to_file
from jmetal.util.termination_criterion import StoppingByEvaluations

if __name__ == "__main__":
    problem = Rastrigin(10)

    algorithm = GDE3(
        problem=problem, population_size=100, cr=0.5, f=0.5, termination_criterion=StoppingByEvaluations(100000)
    )

    algorithm.run()
    front = algorithm.result()

    # Save results to file
    print_function_values_to_file(front, "FUN." + algorithm.get_name() + "." + problem.get_name())
    print_variables_to_file(front, "VAR." + algorithm.get_name() + "." + problem.get_name())

    print("Algorithm (continuous problem): " + algorithm.get_name())
    print("Problem: " + problem.get_name())
    print("Computing time: " + str(algorithm.total_computing_time))
