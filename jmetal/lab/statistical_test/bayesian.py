import numpy as np
import pandas as pd


def bayesian_sign_test(data, rope_limits=[-0.01, 0.01], prior_strength=0.5, prior_place='rope', sample_size=50000,
                       return_sample=False):
    """ Bayesian version of the sign test.

    :param data: An (n x 2) array or DataFrame contaning the results. In data, each column represents an algorithm and, and each row a problem.
    :param rope_limits: array_like. Default [-0.01, 0.01]. Limits of the practical equivalence.
    :param prior_strength: positive float. Default 0.5. Value of the prior strengt
    :param prior_place: string {left, rope, right}. Default 'left'. Place of the pseudo-observation z_0.
    :param sample_size: integer. Default 10000. Total number of random_search samples generated
    :param return_sample: boolean. Default False. If true, also return the samples drawn from the Dirichlet process.

    :return: List of posterior probabilities:
        [Pr(algorith_1 < algorithm_2),
        Pr(algorithm_1 equiv algorithm_2),
        Pr(algorithm_1 > algorithm_2)]
    """

    # Initial Checking
    if type(data) == pd.DataFrame:
        data = data.values

    if data.shape[1] == 2:
        sample1, sample2 = data[:, 0], data[:, 1]
        n = data.shape[0]
    else:
        raise ValueError(
            'Initialization ERROR. Incorrect number of dimensions for axis 1')

    if prior_strength <= 0:
        raise ValueError(
            'Initialization ERROR. prior_strength mustb be a positive float')

    if prior_place not in ['left', 'rope', 'right']:
        raise ValueError(
            'Initialization ERROR. Incorrect value fro prior_place')

    # Compute the differences
    Z = sample1 - sample2

    # Compute the number of pairs diff > right_limit
    Nright = sum(Z > rope_limits[1])
    # Compute the number of pairs diff < right_lelft
    Nleft = sum(Z < rope_limits[0])
    # Compute the number of pairs diff in rope_limits
    Nequiv = n - Nright - Nleft

    # compute the the probabilities that the mean difference of accuracy is in
    # the interval (−Inf, left), [left, right], or (ringth, Inf).

    # Parameters of the Dirichlet distribution
    alpha = np.array([Nleft, Nequiv, Nright], dtype=float) + 1e-6
    alpha[['left', 'rope', 'right'].index(prior_place)] += prior_strength
    # Simulate dirichlet process
    Dprocess = np.random.dirichlet(alpha, sample_size)

    # Compute posterior probabilities
    winner_id = np.argmax(Dprocess, axis=1)
    win_left = sum(winner_id == 0)
    win_rifht = sum(winner_id == 2)
    win_rope = sample_size - win_left - win_rifht

    if return_sample is True:
        return np.array([win_left, win_rope, win_rifht]) / float(sample_size), Dprocess
    else:
        return np.array([win_left, win_rope, win_rifht]) / float(sample_size)


def bayesian_signed_rank_test(data, rope_limits=[-0.01, 0.01], prior_strength=1.0, prior_place='rope',
                              sample_size=10000,
                              return_sample=False):
    """ Bayesian version of the signed rank test.

    :param data: An (n x 2) array or DataFrame contaning the results. In data, each column represents an algorithm and, and each row a problem.
    :param rope_limits: array_like. Default [-0.01, 0.01]. Limits of the practical equivalence.
    :param prior_strength: positive float. Default 0.5. Value of the prior strengt
    :param prior_place: string {left, rope, right}. Default 'left'. Place of the pseudo-observation z_0.
    :param sample_size: integer. Default 10000. Total number of random_search samples generated
    :param return_sample: boolean. Default False. If true, also return the samples drawn from the Dirichlet process.

    :return: List of posterior probabilities:
        [Pr(algorith_1 < algorithm_2), Pr(algorithm_1 equiv algorithm_2), Pr(algorithm_1 > algorithm_2)]
    """

    def weights(n, s):
        alpha = np.ones(n + 1)
        alpha[0] = s
        return np.random.dirichlet(alpha, 1)[0]

    # Initial Checking
    if type(data) == pd.DataFrame:
        data = data.values

    if data.shape[1] == 2:
        sample1, sample2 = data[:, 0], data[:, 1]
        n = data.shape[0]
    else:
        raise ValueError(
            'Initialization ERROR. Incorrect number of dimensions for axis 1')

    if prior_strength <= 0:
        raise ValueError(
            'Initialization ERROR. prior_strength must be a positive float')

    if prior_place not in ['left', 'rope', 'right']:
        raise ValueError(
            'Initialization ERROR. Incorrect value for prior_place')

    # Compute the differences
    Z = sample1 - sample2
    Z0 = [-float('Inf'), 0.0, float('Inf')][['left',
                                             'rope', 'right'].index(prior_place)]
    Z = np.concatenate(([Z0], Z), axis=None)

    # compute the the probabilities that the mean difference of accuracy is in
    # the interval (−Inf, left), [left, right], or (ringth, Inf).

    Dprocess = np.zeros((sample_size, 3))
    for mc in range(sample_size):
        W = weights(n, prior_strength)
        for i in range(n + 1):
            for j in range(i, n + 1):
                aux = Z[i] + Z[j]
                sumval = 2 * (W[i] * W[j]) if i != j else (W[i] * W[j])
                if aux < 2 * rope_limits[0]:
                    Dprocess[mc, 0] += sumval
                elif aux > 2 * rope_limits[1]:
                    Dprocess[mc, 2] += sumval
                else:
                    Dprocess[mc, 1] += sumval

    # Compute posterior probabilities
    winner_id = np.argmax(Dprocess, axis=1)
    win_left = sum(winner_id == 0)
    win_rifht = sum(winner_id == 2)
    win_rope = sample_size - win_left - win_rifht

    if return_sample is True:
        return np.array([win_left, win_rope, win_rifht]) / float(sample_size), Dprocess
    else:
        return np.array([win_left, win_rope, win_rifht]) / float(sample_size)
