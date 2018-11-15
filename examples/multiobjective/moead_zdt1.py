from jmetal.algorithm import MOEAD
from jmetal.problem import ZDT1
from jmetal.operator import Polynomial, DifferentialEvolution

if __name__ == '__main__':
    problem = ZDT1()

    algorithm = MOEAD(
        problem=problem,
        population_size=100,
        max_evaluations=50000,
        delta=0.9,
        nr=2,
        neighbourhood_size=20,
        function_type=MOEAD.AGG,
        crossover=DifferentialEvolution(),
        mutation=Polynomial(probability=1.0 / problem.number_of_variables, distribution_index=20),
    )

    algorithm.run()
    front = algorithm.get_result()

    print(front)
