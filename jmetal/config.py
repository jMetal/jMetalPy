from jmetal.core.observable import DefaultObservable
from jmetal.operator import PolynomialMutation, BitFlipMutation
from jmetal.util.solution_list import RandomGenerator
from jmetal.util.solution_list import SequentialEvaluator
from jmetal.util.termination_criterion import StoppingByEvaluations


class _Store(object):

    def __init__(self):
        super(_Store, self).__init__()
        self.default_observable = DefaultObservable()
        self.default_evaluator = SequentialEvaluator()
        self.default_generator = RandomGenerator()
        self.default_termination_criteria = StoppingByEvaluations(max=25000)
        self.default_mutation = {
            'real': PolynomialMutation(probability=0.15, distribution_index=20),
            'binary': BitFlipMutation(0.15)
        }


store = _Store()
