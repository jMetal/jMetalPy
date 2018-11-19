from jmetal.core.observable import DefaultObservable
from jmetal.component import SequentialEvaluator, RandomGenerator

store = {
    'default_observable': DefaultObservable(),
    'default_evaluator': SequentialEvaluator(),
    'default_generator': RandomGenerator()
}