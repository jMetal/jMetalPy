"""Regression tests for rng support in the Bayesian statistical tests.

Both bayesian_sign_test() and bayesian_signed_rank_test() previously drew their
Dirichlet-process samples from the global numpy.random state unconditionally.
"""

import numpy as np

from jmetal.lab.statistical_test.bayesian import bayesian_sign_test, bayesian_signed_rank_test


def _sample_data():
    rng = np.random.default_rng(0)
    return rng.normal(size=(20, 2))


class TestBayesianSignTest:
    def test_rng_produces_deterministic_results_for_a_fixed_seed(self):
        data = _sample_data()

        result_a = bayesian_sign_test(data, sample_size=500, rng=np.random.default_rng(42))
        result_b = bayesian_sign_test(data, sample_size=500, rng=np.random.default_rng(42))

        assert (result_a == result_b).all()

    def test_without_rng_still_returns_valid_probabilities(self):
        data = _sample_data()

        result = bayesian_sign_test(data, sample_size=500)

        assert len(result) == 3
        assert abs(sum(result) - 1.0) < 1e-9


class TestBayesianSignedRankTest:
    def test_rng_produces_deterministic_results_for_a_fixed_seed(self):
        data = _sample_data()

        result_a = bayesian_signed_rank_test(
            data, sample_size=50, rng=np.random.default_rng(42)
        )
        result_b = bayesian_signed_rank_test(
            data, sample_size=50, rng=np.random.default_rng(42)
        )

        assert (result_a == result_b).all()

    def test_without_rng_still_returns_valid_probabilities(self):
        data = _sample_data()

        result = bayesian_signed_rank_test(data, sample_size=50)

        assert len(result) == 3
        assert abs(sum(result) - 1.0) < 1e-9
