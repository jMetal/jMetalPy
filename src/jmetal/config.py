from jmetal.core.observer import Observable
from jmetal.operator import BitFlipMutation, PolynomialMutation
from jmetal.util.comparator import DominanceComparator
from jmetal.util.evaluator import Evaluator, SequentialEvaluator
from jmetal.util.generator import RandomGenerator
from jmetal.util.observable import DefaultObservable
from jmetal.util.termination_criterion import StoppingByEvaluations


class _Store:
    @property
    def default_observable(self) -> Observable:
        return DefaultObservable()

    @property
    def default_evaluator(self) -> Evaluator:
        return SequentialEvaluator()

    @property
    def default_generator(self):
        return RandomGenerator()

    @property
    def default_termination_criteria(self):
        return StoppingByEvaluations(max_evaluations=25000)

    @property
    def default_comparator(self):
        return DominanceComparator()

    @property
    def default_mutation(self):
        return {"real": PolynomialMutation(probability=0.15, distribution_index=20), "binary": BitFlipMutation(0.15)}


store = _Store()
