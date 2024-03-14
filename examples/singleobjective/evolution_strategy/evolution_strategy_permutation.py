from jmetal.algorithm.singleobjective.evolution_strategy import EvolutionStrategy
from jmetal.operator.mutation import PermutationSwapMutation
from jmetal.problem.singleobjective.tsp import TSP
from jmetal.util.termination_criterion import StoppingByTime

if __name__ == "__main__":
    problem = TSP(instance="resources/TSP_instances/kroA100.tsp")

    print(f"Solving TSP problem with {problem.number_of_cities} cities.")

    algorithm = EvolutionStrategy(
        problem=problem,
        mu=1,
        lambda_=10,
        mutation=PermutationSwapMutation(1.0 / problem.number_of_cities),
        elitist=False,
        termination_criterion=StoppingByTime(10),
    )

    algorithm.run()
    result = algorithm.result()

    print(f"Algorithm: {algorithm.get_name()}")
    print(f"Solution: {result.variables}")
    print(f"The shortest path length: {result.objectives[0]}")
    print(f"Computing time: {algorithm.total_computing_time}")
    print(f"Number of evaluations: {algorithm.evaluations}")
