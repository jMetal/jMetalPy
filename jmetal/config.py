from jmetal.component import SequentialEvaluator, RandomGenerator, DefaultObservable


class _Store(object):

    def __init__(self):
        super(_Store, self).__init__()
        self.default_observable = DefaultObservable()
        self.default_evaluator = SequentialEvaluator()
        self.default_generator = RandomGenerator()


store = _Store()
