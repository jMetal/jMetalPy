from jmetal.component import SequentialEvaluator, RandomGenerator, DefaultObservable
from jmetal.util.termination_criteria import StoppingByEvaluations


class _Store(object):

    def __init__(self):
        super(_Store, self).__init__()
        self.default_observable = DefaultObservable()
        self.default_evaluator = SequentialEvaluator()
        self.default_generator = RandomGenerator()
        self.default_termination_criteria = StoppingByEvaluations(max=25000)


store = _Store()
