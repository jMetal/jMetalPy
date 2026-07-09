from jmetal.algorithm.singleobjective.genetic_algorithm import GeneticAlgorithm
from jmetal.operator.crossover import SBXCrossover
from jmetal.operator.mutation import PolynomialMutation
from jmetal.problem.singleobjective.unconstrained import Rastrigin
from jmetal.util.observer import PrintObjectivesObserver
from jmetal.util.termination_criterion import StoppingByEvaluations

if __name__ == "__main__":
    problem = Rastrigin(10)

    algorithm = GeneticAlgorithm(
        problem=problem,
        population_size=100,
        offspring_population_size=1,
        mutation=PolynomialMutation(1.0 / problem.number_of_variables(), 20.0),
        crossover=SBXCrossover(0.9, 5.0),
        termination_criterion=StoppingByEvaluations(max_evaluations=150000),
    )

    algorithm.observable.register(observer=PrintObjectivesObserver(5000))

    algorithm.run()
    result = algorithm.result()

    print(f"Algorithm: {algorithm.get_name()}")
    print(f"Problem: {problem.name()}")
    print(f"Solution: {result.variables}")
    print(f"Fitness: {result.objectives[0]}")
    print(f"Computing time: {algorithm.total_computing_time}")
