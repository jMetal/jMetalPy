# -*- coding: utf-8 -*-
"""

TEST jMetal

@author: María Fdez Hijano

"""
 
from jmetal.algorithm.multiobjective import NSGAII
from jmetal.operator import PMXCrossover, PermutationSwapMutation
from jmetal.problem.multiobjective.tsp import TSP
from jmetal.util.termination_criterion import StoppingByEvaluations


problem = TSP('resources/TSP_instances/test.tsp','resources/TSP_instances/test1.tsp')


max_evaluations = 5000

algorithm = NSGAII(
    problem=problem,
    population_size=100,
    offspring_population_size=100,
    mutation=PermutationSwapMutation(probability=0.2),
    crossover=PMXCrossover(probability=0.9),
    termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations)
)

algorithm.run()

from jmetal.util.solution import get_non_dominated_solutions, print_function_values_to_file, print_variables_to_file

front = get_non_dominated_solutions(algorithm.get_result())
# save to files
print_function_values_to_file(front, 'output/tmp/FUN.NSGAII.TSPMULTI')
print_variables_to_file(front, 'output/tmp/VAR.NSGAII.TSPMULTI')


from jmetal.lab.visualization import Plot

plot_front = Plot(title='Pareto front approximation', axis_labels=['x', 'y'])
plot_front.plot(front, label='NSGAII-TSPMULTI', filename='output/tmp/NSGAII-TSPMULTI', format='png')

