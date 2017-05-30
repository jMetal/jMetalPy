from jmetal.algorithm.singleobjective.evolutionaryalgorithm.generationalGeneticAlgorithm import GenerationalGeneticAlgorithm
from jmetal.core.solution.binarySolution import BinarySolution
from jmetal.core.solution.floatSolution import FloatSolution
from jmetal.core.util.observer.observer import Observer
from jmetal.operator.crossover.SBX import SBX
from jmetal.operator.crossover.singlepoint import SinglePoint
from jmetal.operator.mutation.bitflip import BitFlip
from jmetal.operator.mutation.polynomial import Polynomial
from jmetal.operator.selection.binarytournament import BinaryTournament
from jmetal.problem.singleobjective.onemax import OneMax
from jmetal.problem.singleobjective.sphere import Sphere


def main():
    float_example()

class AlgorithmObserver(Observer):
    def update(self, *args, **kwargs):
        print("Evaluations: " + str(kwargs["evaluations"]))
        print("Evaluations: " + str(kwargs["best"].objectives[0]))


def float_example() -> None:
    variables = 10
    problem = Sphere(variables)
    algorithm = GenerationalGeneticAlgorithm[FloatSolution, FloatSolution](
        problem,
        population_size = 100,
        max_evaluations = 25000,
        mutation_operator = Polynomial(1.0/variables, distribution_index=20),
        crossover_operator = SBX(1.0, distribution_index=20),
        selection_operator = BinaryTournament())

    observer = AlgorithmObserver()

    algorithm.observable.register(observer=observer)

    algorithm.start()
    algorithm.join()

    result = algorithm.get_result()
    print("Algorithm: " + algorithm.get_name())
    print("Problem: " + problem.get_name())
    print("Solution: " + str(result.variables))
    print("Fitness:  " + str(result.objectives[0]))

if __name__ == '__main__':
    main()
