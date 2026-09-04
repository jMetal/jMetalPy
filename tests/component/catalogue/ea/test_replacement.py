"""Tests for the Replacement component re-export."""

from jmetal.component.catalogue.ea import replacement as component_replacement
from jmetal.operator import replacement as operator_replacement


class TestReplacementReExport:
    def test_re_exports_the_same_replacement_abc(self):
        # No adapter needed: the operator-level Replacement ABC already matches the
        # component contract, so the catalogue must not define a second one.
        assert component_replacement.Replacement is operator_replacement.Replacement

    def test_re_exports_the_same_ranking_and_density_estimator_replacement(self):
        assert (
            component_replacement.RankingAndDensityEstimatorReplacement
            is operator_replacement.RankingAndDensityEstimatorReplacement
        )

    def test_re_exports_the_same_removal_policy_type(self):
        assert component_replacement.RemovalPolicyType is operator_replacement.RemovalPolicyType
