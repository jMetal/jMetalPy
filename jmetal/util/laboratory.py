from typing import List, Type

from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.algorithm.multiobjective.smpso import SMPSO
from jmetal.core.algorithm import Algorithm
from jmetal.core.problem import Problem
from jmetal.problem.multiobjective.zdt import ZDT1
from jmetal.util.quality_indicator import HyperVolume

"""
.. module:: laboratory
   :platform: Unix, Windows
   :synopsis: Run experiments.

.. moduleauthor:: Antonio Benítez-Hidalgo <antonio.b@uma.es>
"""


def experiment(algorithm_list: List[Type[Algorithm]], problem_list: List[Type[Problem]], metric_list: list):
    result = dict()

    for problem in problem_list:
        if isinstance(problem, type):
            problem = problem()

        for algorithm in algorithm_list:
            if isinstance(algorithm, type):
                algorithm = algorithm(problem=problem)

            algorithm.run()
            solution_list = algorithm.get_result()

            result[algorithm.get_name()][problem.get_name()] = [metric.compute(solution_list) for metric in metric_list]

    print(result)


if __name__ == '__main__':
    algorithm = [NSGAII, SMPSO]
    problem = [ZDT1]
    metric = [HyperVolume(referencePoint=[1, 1])]

    experiment(algorithm, problem, metric)
