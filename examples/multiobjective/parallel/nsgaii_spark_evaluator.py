from examples.multiobjective.parallel.zdt1_modified import ZDT1Modified
from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.operator import SBXCrossover, PolynomialMutation
from jmetal.util.observer import ProgressBarObserver
from jmetal.util.solutions.evaluator import SparkEvaluator
from jmetal.util.termination_criterion import StoppingByEvaluations

if __name__ == '__main__':
    problem = ZDT1Modified()

    max_evaluations = 100

    algorithm = NSGAII(
        problem=problem,
        population_size=10,
        offspring_population_size=10,
        mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20),
        crossover=SBXCrossover(probability=1.0, distribution_index=20),
        population_evaluator=SparkEvaluator(),
        termination_criterion=StoppingByEvaluations(max=max_evaluations)
    )

    algorithm.observable.register(observer=ProgressBarObserver(max=max_evaluations))

    algorithm.run()
    front = algorithm.get_result()

    print('Algorithm (continuous problem): ' + algorithm.get_name())
    print('Problem: ' + problem.get_name())
    print('Computing time: ' + str(algorithm.total_computing_time))
