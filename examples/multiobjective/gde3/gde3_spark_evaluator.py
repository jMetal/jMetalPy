from jmetal.algorithm.multiobjective.gde3 import GDE3
from jmetal.problem.multiobjective.zdt import ZDT1Modified
from jmetal.util.evaluator import SparkEvaluator
from jmetal.util.solution import print_function_values_to_file, print_variables_to_file
from jmetal.util.termination_criterion import StoppingByEvaluations

if __name__ == '__main__':
    problem = ZDT1Modified()

    algorithm = GDE3(
        problem=problem,
        population_size=10,
        cr=0.5,
        f=0.5,
        termination_criterion=StoppingByEvaluations(max_evaluations=100),
        population_evaluator=SparkEvaluator()
    )

    algorithm.run()
    front = algorithm.get_result()

    # Save results to file
    print_function_values_to_file(front, 'FUN.' + algorithm.get_name() + "." + problem.get_name())
    print_variables_to_file(front, 'VAR.' + algorithm.get_name() + "." + problem.get_name())

    print('Algorithm (continuous problem): ' + algorithm.get_name())
    print('Problem: ' + problem.get_name())
    print('Computing time: ' + str(algorithm.total_computing_time))
