from jmetal.algorithm.singleobjective.genetic_algorithm import GeneticAlgorithm
from jmetal.operator import BitFlipMutation, SPXCrossover
from jmetal.problem import OneMax
from jmetal.util.observer import PrintObjectivesObserver
from jmetal.util.termination_criterion import StoppingByEvaluations

if __name__ == "__main__":
    problem = OneMax(number_of_bits=512)

    algorithm = GeneticAlgorithm(
        problem=problem,
        population_size=40,
        offspring_population_size=40,
        mutation=BitFlipMutation(1.0 / problem.total_number_of_bits()),
        crossover=SPXCrossover(1.0),
        termination_criterion=StoppingByEvaluations(max_evaluations=20000),
    )

    algorithm.observable.register(observer=PrintObjectivesObserver(100))

    algorithm.run()
    result = algorithm.result()

    print("Algorithm: {}".format(algorithm.get_name()))
    print("Problem: {}".format(problem.name()))
    print("Solution: " + result.get_binary_string())
    print("Fitness:  " + str(result.objectives[0]))
    print("Computing time: {}".format(algorithm.total_computing_time))
