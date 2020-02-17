import threading
from abc import ABC, abstractmethod

from jmetal.core.observer import Observer
from jmetal.core.quality_indicator import QualityIndicator

"""
.. module:: termination_criterion
   :platform: Unix, Windows
   :synopsis: Implementation of stopping conditions.

.. moduleauthor:: Antonio Benítez-Hidalgo <antonio.b@uma.es>
"""


class TerminationCriterion(Observer, ABC):

    @abstractmethod
    def update(self, *args, **kwargs):
        pass

    @property
    @abstractmethod
    def is_met(self):
        pass


class StoppingByEvaluations(TerminationCriterion):

    def __init__(self, max_evaluations: int):
        super(StoppingByEvaluations, self).__init__()
        self.max_evaluations = max_evaluations
        self.evaluations = 0

    def update(self, *args, **kwargs):
        self.evaluations = kwargs['EVALUATIONS']

    @property
    def is_met(self):
        return self.evaluations >= self.max_evaluations


class StoppingByTime(TerminationCriterion):

    def __init__(self, max_seconds: int):
        super(StoppingByTime, self).__init__()
        self.max_seconds = max_seconds
        self.seconds = 0.0

    def update(self, *args, **kwargs):
        self.seconds = kwargs['COMPUTING_TIME']

    @property
    def is_met(self):
        return self.seconds >= self.max_seconds


def key_has_been_pressed(stopping_by_keyboard):
    input('PRESS ANY KEY + ENTER: ')
    stopping_by_keyboard.key_pressed = True


class StoppingByKeyboard(TerminationCriterion):

    def __init__(self):
        super(StoppingByKeyboard, self).__init__()
        self.key_pressed = False
        thread = threading.Thread(target=key_has_been_pressed, args=(self,))
        thread.start()

    def update(self, *args, **kwargs):
        pass

    @property
    def is_met(self):
        return self.key_pressed


class StoppingByQualityIndicator(TerminationCriterion):

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
