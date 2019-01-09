from jmetal.core.observable import DefaultObservable

from jmetal.operator import Polynomial, BitFlip
from jmetal.util.evaluator import SequentialEvaluator
from jmetal.util.generator import RandomGenerator

from jmetal.util.termination_criterion import StoppingByEvaluations


class _Store(object):

    def __init__(self):
        super(_Store, self).__init__()
        self.default_observable = DefaultObservable()
        self.default_evaluator = SequentialEvaluator()
        self.default_generator = RandomGenerator()
        self.default_termination_criteria = StoppingByEvaluations(max=25000)
        self.default_mutation = {
            'real': Polynomial(probability=0.15, distribution_index=20),
            'binary': BitFlip(0.15)
        }


store = _Store()
