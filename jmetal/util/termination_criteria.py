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

    @abstractmethod
    def update(self, *args, **kwargs):
        pass

    @property
    @abstractmethod
    def is_met(self):
        pass


class StoppingByEvaluations(TerminationCriteria):

    def __init__(self, max: int):
        super(StoppingByEvaluations, self).__init__()
        self.max_evaluations = max
        self.evaluations = 0

    def update(self, *args, **kwargs):
        self.evaluations = kwargs['EVALUATIONS']

    @property
    def is_met(self):
        return self.evaluations >= self.max_evaluations


class StoppingByTime(TerminationCriteria):

    def __init__(self, max_seconds: int):
        super(StoppingByTime, self).__init__()
        self.max_seconds = max_seconds
        self.seconds = 0.0

    def update(self, *args, **kwargs):
        self.seconds = kwargs['COMPUTING_TIME']

    @property
    def is_met(self):
        return self.seconds >= self.max_seconds


class StoppingByQualityIndicator(TerminationCriteria):

    def __init__(self, quality_indicator: QualityIndicator, expected_value: float, degree: float):
        super(StoppingByQualityIndicator, self).__init__()
        self.quality_indicator = quality_indicator
        self.expected_value = expected_value
        self.degree = degree
        self.value = 0.0

    def update(self, *args, **kwargs):
        solutions = kwargs['SOLUTIONS']

        if solutions:
            self.value = self.quality_indicator.compute(solutions)

    @property
    def is_met(self):
        if self.quality_indicator.is_minimization:
            met = self.value * self.degree < self.expected_value
        else:
            met = self.value * self.degree > self.expected_value

        return met
