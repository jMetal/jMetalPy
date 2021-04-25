# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 17:46:16 2020

@author: María Fdez Hijano
"""
from jmetal.algorithm.singleobjective.genetic_algorithm import GeneticAlgorithm
from jmetal.operator import BinaryTournamentSelection
from jmetal.operator.crossover import PMXCrossover
from jmetal.operator.mutation import PermutationSwapMutation
from jmetal.util.comparator import MultiComparator
from jmetal.util.density_estimator import CrowdingDistance
from jmetal.util.ranking import FastNonDominatedRanking
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.problem.singleobjective.cvrp import CVRP


problem = CVRP('resources/CVRP_instances/A-n80-k10-dummy_points.vrp',10, False)

algorithm = GeneticAlgorithm(
        problem=problem,
        population_size=88,
        offspring_population_size=88,
        mutation=PermutationSwapMutation(0.2),
        crossover=PMXCrossover(0.9),
        selection=BinaryTournamentSelection(
            MultiComparator([FastNonDominatedRanking.get_comparator(),
                             CrowdingDistance.get_comparator()])),
        termination_criterion=StoppingByEvaluations(max_evaluations=150000)
    )

algorithm.run()
result = algorithm.get_result()
print('Algorithm: {}'.format(algorithm.get_name()))
print('Problem: {}'.format(problem.get_name()))
print('Solution: {}'.format(result.variables))
print('Fitness: {}'.format(result.objectives[0]))
print('Computing time: {}'.format(algorithm.total_computing_time))
