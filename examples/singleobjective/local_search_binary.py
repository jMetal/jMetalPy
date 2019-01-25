from jmetal.algorithm.singleobjective.local_search import LocalSearch
from jmetal.operator import BitFlipMutation
from jmetal.problem import OneMax
from jmetal.util.observer import ProgressBarObserver
from jmetal.util.termination_criterion import StoppingByEvaluations

if __name__ == '__main__':
    problem = OneMax(number_of_bits=512)

    algorithm = LocalSearch(
        problem=problem,
        mutation=BitFlipMutation(probability=1.0 / problem.number_of_bits),
        termination_criterion=StoppingByEvaluations(max=25000)
    )

    progress_bar = ProgressBarObserver(max=25000)
    algorithm.observable.register(observer=progress_bar)

    algorithm.run()
    result = algorithm.get_result()

    print('Algorithm: ' + algorithm.get_name())
    print('Problem: ' + problem.get_name())
    print('Solution: ' + str(result.variables[0]))
    print('Fitness:  ' + str(result.objectives[0]))
    print('Computing time: ' + str(algorithm.total_computing_time))
