from abc import ABCMeta, abstractmethod

from jmetal.component.quality_indicator import QualityIndicator
from jmetal.util.observable import Observer

"""
.. module:: termination_criteria
   :platform: Unix, Windows
   :synopsis: Implementation of stopping conditions.

.. moduleauthor:: Antonio Benítez-Hidalgo <antonio.b@uma.es>
"""


class TerminationCriteria(Observer):

    __metaclass__ = ABCMeta

    def __init__(self):
        self.is_met = False

    @abstractmethod
    def update(self, *args, **kwargs):
        pass


class StoppingByEvaluations(TerminationCriteria):

    def __init__(self, max: int):
        super(StoppingByEvaluations, self).__init__()
        self.max_evaluations = max

    def update(self, *args, **kwargs):
        self.is_met = kwargs['EVALUATIONS'] >= self.max_evaluations


class StoppingByTime(TerminationCriteria):

    def __init__(self, max_seconds: int):
        super(StoppingByTime, self).__init__()
        self.max_seconds = max_seconds

    def update(self, *args, **kwargs):
        self.is_met = kwargs['COMPUTING_TIME'] >= self.max_seconds


class StoppingByQualityIndicator(TerminationCriteria):

    def __init__(self, quality_indicator: QualityIndicator, expected_value: float, degree: float):
        super(StoppingByQualityIndicator, self).__init__()
        self.quality_indicator = quality_indicator
        self.expected_value = expected_value
        self.degree = degree

    def update(self, *args, **kwargs):
        solutions = kwargs['POPULATION']
        value = self.quality_indicator.compute(solutions)

        if self.quality_indicator.is_minimization:
            self.is_met = value * self.degree < self.expected_value
        else:
            self.is_met = value * self.degree > self.expected_value
