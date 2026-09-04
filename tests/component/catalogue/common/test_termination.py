"""Tests for the Termination component."""

from jmetal.component.catalogue.common.termination import (
    Termination,
    TerminationByEvaluations,
)


class TestTerminationByEvaluations:
    def test_is_not_met_before_reaching_the_maximum(self):
        termination = TerminationByEvaluations(max_evaluations=100)

        assert termination.is_met({"EVALUATIONS": 50}) is False

    def test_is_met_once_the_maximum_is_reached(self):
        termination = TerminationByEvaluations(max_evaluations=100)

        assert termination.is_met({"EVALUATIONS": 100}) is True

    def test_is_met_when_the_maximum_is_exceeded(self):
        termination = TerminationByEvaluations(max_evaluations=100)

        assert termination.is_met({"EVALUATIONS": 150}) is True

    def test_satisfies_the_termination_protocol(self):
        assert isinstance(TerminationByEvaluations(100), Termination)

    def test_a_plain_function_wrapped_in_an_object_satisfies_the_protocol(self):
        class AlwaysMet:
            def is_met(self, status):
                return True

        assert isinstance(AlwaysMet(), Termination)
