from jmetal.algorithm.singleobjective.genetic_algorithm import GeneticAlgorithm
from jmetal.operator import SPXCrossover, BitFlipMutation, BinaryTournamentSelection
from jmetal.problem import OneMax
from jmetal.util.observer import PrintObjectivesObserver
from jmetal.util.termination_criterion import StoppingByEvaluations

if __name__ == '__main__':
<<<<<<< HEAD
    problem = OneMax(number_of_bits=1024)

    algorithm = GeneticAlgorithm(
        problem=problem,
        population_size=100,
        offspring_population_size=100,
=======
    problem = OneMax(number_of_bits=512)

    algorithm = GeneticAlgorithm(
        problem=problem,
        population_size=40,
        offspring_population_size=40,
>>>>>>> 52e0b172f0c6d651ba08b961a90a382f0a4b8e0f
        mutation=BitFlipMutation(1.0 / problem.number_of_bits),
        crossover=SPXCrossover(1.0),
        selection=BinaryTournamentSelection(),
        termination_criterion=StoppingByEvaluations(max=20000)
    )

    algorithm.observable.register(observer=PrintObjectivesObserver(frequency=1000))

    algorithm.run()
    result = algorithm.get_result()

    print('Algorithm: {}'.format(algorithm.get_name()))
    print('Problem: {}'.format(problem.get_name()))
    print('Solution: ' + result.get_binary_string())
    print('Fitness:  ' + str(result.objectives[0]))
    print('Computing time: {}'.format(algorithm.total_computing_time))
