from scipy.stats import chi2, f, binom, norm

from jmetal.lab.statistical_test.apv_procedures import *


def ranks(data: np.array, descending=False):
    """ Computes the rank of the elements in data.

    :param data: 2-D matrix
    :param descending: boolean (default False). If true, rank is sorted in descending order.
    :return: ranks, where ranks[i][j] == rank of the i-th row w.r.t the j-th column.
    """
    s = 0 if (descending is False) else 1

    # Compute ranks. (ranks[i][j] == rank of the i-th treatment on the j-th sample.)
    if data.ndim == 2:
        ranks = np.ones(data.shape)
        for i in range(data.shape[0]):
            values, indices, rep = np.unique(
                (-1) ** s * np.sort((-1) ** s * data[i, :]), return_index=True, return_counts=True, )
            for j in range(data.shape[1]):
                ranks[i, j] += indices[values == data[i, j]] + \
                               0.5 * (rep[values == data[i, j]] - 1)
        return ranks
    elif data.ndim == 1:
        ranks = np.ones((data.size,))
        values, indices, rep = np.unique(
            (-1) ** s * np.sort((-1) ** s * data), return_index=True, return_counts=True, )
        for i in range(data.size):
            ranks[i] += indices[values == data[i]] + \
                        0.5 * (rep[values == data[i]] - 1)
        return ranks


def sign_test(data):
    """ Given the results drawn from two algorithms/methods X and Y, the sign test analyses if
    there is a difference between X and Y.

    .. note:: Null Hypothesis: Pr(X<Y)= 0.5

    :param data: An (n x 2) array or DataFrame contaning the results. In data, each column represents an algorithm and, and each row a problem.
    :return p_value: The associated p-value from the binomial distribution.
    :return bstat: Number of successes.
    """

    if type(data) == pd.DataFrame:
        data = data.values

    if data.shape[1] == 2:
        X, Y = data[:, 0], data[:, 1]
        n_perf = data.shape[0]
    else:
        raise ValueError(
            'Initialization ERROR. Incorrect number of dimensions for axis 1')

    # Compute the differences
    Z = X - Y
    # Compute the number of pairs Z<0
    Wminus = sum(Z < 0)
    # If H_0 is true ---> W follows Binomial(n,0.5)
    p_value_minus = 1 - binom.cdf(k=Wminus, p=0.5, n=n_perf)

    # Compute the number of pairs Z>0
    Wplus = sum(Z > 0)
    # If H_0 is true ---> W follows Binomial(n,0.5)
    p_value_plus = 1 - binom.cdf(k=Wplus, p=0.5, n=n_perf)

    p_value = 2 * min([p_value_minus, p_value_plus])

    return pd.DataFrame(data=np.array([Wminus, Wplus, p_value]), index=['Num X<Y', 'Num X>Y', 'p-value'],
                        columns=['Results'])


def friedman_test(data):
    """ Friedman ranking test.

    ..note:: Null Hypothesis: In a set of k (>=2) treaments (or tested algorithms), all the treatments are equivalent, so their average ranks should be equal.

    :param data: An (n x 2) array or DataFrame contaning the results. In data, each column represents an algorithm and, and each row a problem.
    :return p_value: The associated p-value.
    :return friedman_stat: Friedman's chi-square.
    """

    # Initial Checking
    if type(data) == pd.DataFrame:
        data = data.values

    if data.ndim == 2:
        n_samples, k = data.shape
    else:
        raise ValueError(
            'Initialization ERROR. Incorrect number of array dimensions')
    if k < 2:
        raise ValueError(
            'Initialization Error. Incorrect number of dimensions for axis 1.')

    # Compute ranks.
    datarank = ranks(data)

    # Compute for each algorithm the ranking average.
    avranks = np.mean(datarank, axis=0)

    # Get Friedman statistics
    friedman_stat = (12.0 * n_samples) / (k * (k + 1.0)) * \
                    (np.sum(avranks ** 2) - (k * (k + 1) ** 2) / 4.0)

    # Compute p-value
    p_value = (1.0 - chi2.cdf(friedman_stat, df=(k - 1)))

    return pd.DataFrame(data=np.array([friedman_stat, p_value]), index=['Friedman-statistic', 'p-value'],
                        columns=['Results'])


def friedman_aligned_rank_test(data):
    """ Method of aligned ranks for the Friedman test.

    ..note:: Null Hypothesis: In a set of k (>=2) treaments (or tested algorithms), all the treatments are equivalent, so their average ranks should be equal.

    :param data: An (n x 2) array or DataFrame contaning the results. In data, each column represents an algorithm and, and each row a problem.
    :return p_value: The associated p-value.
    :return aligned_rank_stat: Friedman's aligned rank chi-square statistic.
    """

    # Initial Checking
    if type(data) == pd.DataFrame:
        data = data.values

    if data.ndim == 2:
        n_samples, k = data.shape
    else:
        raise ValueError(
            'Initialization ERROR. Incorrect number of array dimensions')
    if k < 2:
        raise ValueError(
            'Initialization Error. Incorrect number of dimensions for axis 1.')

    # Compute the average value achieved by all algorithms in each problem
    control = np.mean(data, axis=1)
    # Compute the difference between control an data
    diff = [data[:, j] - control for j in range(data.shape[1])]
    # rank diff
    alignedRanks = ranks(np.ravel(diff))
    alignedRanks = np.reshape(alignedRanks, newshape=(n_samples, k), order='F')

    # Compute statistic
    Rhat_i = np.sum(alignedRanks, axis=1)
    Rhat_j = np.sum(alignedRanks, axis=0)
    si, sj = np.sum(Rhat_i ** 2), np.sum(Rhat_j ** 2)

    A = sj - (k * n_samples ** 2 / 4.0) * (k * n_samples + 1) ** 2
    B1 = (k * n_samples * (k * n_samples + 1) * (2 * k * n_samples + 1) / 6.0)
    B2 = si / float(k)

    alignedRanks_stat = ((k - 1) * A) / (B1 - B2)

    p_value = 1 - chi2.cdf(alignedRanks_stat, df=k - 1)

    return pd.DataFrame(data=np.array([alignedRanks_stat, p_value]), index=['Aligned Rank stat', 'p-value'],
                        columns=['Results'])


def quade_test(data):
    """ Quade test.

    ..note:: Null Hypothesis: In a set of k (>=2) treaments (or tested algorithms), all the treatments are equivalent, so their average ranks should be equal.

    :param data: An (n x 2) array or DataFrame contaning the results. In data, each column represents an algorithm and, and each row a problem.
    :return p_value: The associated p-value from the F-distribution.
    :return fq: Computed F-value.
    """

    # Initial Checking
    if type(data) == pd.DataFrame:
        data = data.values

    if data.ndim == 2:
        n_samples, k = data.shape
    else:
        raise ValueError(
            'Initialization ERROR. Incorrect number of array dimensions')
    if k < 2:
        raise ValueError(
            'Initialization Error. Incorrect number of dimensions for axis 1.')

    # Compute ranks.
    datarank = ranks(data)
    # Compute the range of each problem
    problemRange = np.max(data, axis=1) - np.min(data, axis=1)
    # Compute problem rank
    problemRank = ranks(problemRange)

    # Compute S_stat: weight of each observation within the problem, adjusted to reflect
    # the significance of the problem when it appears.
    S_stat = np.zeros((n_samples, k))
    for i in range(n_samples):
        S_stat[i, :] = problemRank[i] * (datarank[i, :] - 0.5 * (k + 1))
    Salg = np.sum(S_stat, axis=0)

    # Compute Fq (Quade Test statistic) and associated p_value
    A = np.sum(S_stat ** 2)
    B = np.sum(Salg ** 2) / float(n_samples)

    if A == B:
        Fq = np.Inf
        p_value = (1 / (np.math.factorial(k))) ** (n_samples - 1)
    else:
        Fq = (n_samples - 1.0) * B / (A - B)
        p_value = 1 - f.cdf(Fq, k - 1, (k - 1) * (n_samples - 1))

    return pd.DataFrame(data=np.array([Fq, p_value]), index=['Quade Test statistic', 'p-value'], columns=['Results'])


def friedman_ph_test(data, control=None, apv_procedure=None):
    """ Friedman post-hoc test.

    :param data: An (n x 2) array or DataFrame contaning the results. In data, each column represents an algorithm and, and each row a problem.
    :param control: optional int or string. Default None. Index or Name of the control algorithm. If control = None all FriedmanPosHocTest considers all possible comparisons among algorithms.
    :param apv_procedure: optional string. Default None.
        Name of the procedure for computing adjusted p-values. If apv_procedure
        is None, adjusted p-value are not computed, else the values are computed
        according to the specified procedure:
        For 1 vs all comparisons.
            {'Bonferroni', 'Holm', 'Hochberg', 'Holland', 'Finner', 'Li'}
        For all vs all coparisons.
            {'Shaffer', 'Holm', 'Nemenyi'}

    :return z_values: Test statistic.
    :return p_values: The p-value according to the Studentized range distribution.
    """

    # Initial Checking
    if type(data) == pd.DataFrame:
        algorithms = data.columns
        data = data.values
    elif type(data) == np.ndarray:
        algorithms = np.array(['Alg%d' % alg for alg in range(data.shape[1])])

    if control is None:
        index = algorithms
    elif type(control) == int:
        index = [algorithms[control]]
    else:
        index = [control]

    if data.ndim == 2:
        n_samples, k = data.shape
    else:
        raise ValueError(
            'Initialization ERROR. Incorrect number of array dimensions.')
    if k < 2:
        raise ValueError(
            'Initialization Error. Incorrect number of dimensions for axis 1.')

    if control is not None:
        if type(control) == int and control >= data.shape[1]:
            raise ValueError('Initialization ERROR. control is out of bounds')
        if type(control) == str and control not in algorithms:
            raise ValueError(
                'Initialization ERROR. %s is not a column name of data' % control)

    if apv_procedure is not None:
        if apv_procedure not in ['Bonferroni', 'Holm', 'Hochberg', 'Hommel', 'Holland', 'Finner', 'Li', 'Shaffer',
                                 'Nemenyi']:
            raise ValueError(
                'Initialization ERROR. Incorrect value for APVprocedure.')

    # Compute ranks.
    datarank = ranks(data)
    # Compute for each algorithm the ranking average.
    avranks = np.mean(datarank, axis=0)

    # Compute z-values
    aux = np.sqrt((k * (k + 1)) / (6.0 * n_samples))

    if control is None:
        z = np.zeros((k, k))
        for i in range(k):
            for j in range(i + 1, k):
                z[i, j] = abs(avranks[i] - avranks[j]) / aux
        z += z.T
    else:
        if type(control) == str:
            control = int(np.where(algorithms == control)[0])
        z = np.zeros((1, k))
        for j in range(k):
            z[0, j] = abs(avranks[control] - avranks[j]) / aux

    # Compute associated p-value
    p_value = 2 * (1.0 - norm.cdf(z))

    pvalues_df = pd.DataFrame(data=p_value, index=index, columns=algorithms)
    zvalues_df = pd.DataFrame(data=z, index=index, columns=algorithms)

    if apv_procedure is None:
        return zvalues_df, pvalues_df
    else:
        if apv_procedure == 'Bonferroni':
            ap_vs_df = bonferroni_dunn(pvalues_df, control=control)
        elif apv_procedure == 'Holm':
            ap_vs_df = holm(pvalues_df, control=control)
        elif apv_procedure == 'Hochberg':
            ap_vs_df = hochberg(pvalues_df, control=control)
        elif apv_procedure == 'Holland':
            ap_vs_df = holland(pvalues_df, control=control)
        elif apv_procedure == 'Finner':
            ap_vs_df = finner(pvalues_df, control=control)
        elif apv_procedure == 'Li':
            ap_vs_df = li(pvalues_df, control=control)
        elif apv_procedure == 'Shaffer':
            ap_vs_df = shaffer(pvalues_df)
        elif apv_procedure == 'Nemenyi':
            ap_vs_df = nemenyi(pvalues_df)

        return zvalues_df, pvalues_df, ap_vs_df


def friedman_aligned_ph_test(data, control=None, apv_procedure=None):
    """ Friedman Aligned Ranks post-hoc test.

    :param data: An (n x 2) array or DataFrame contaning the results. In data, each column represents an algorithm and, and each row a problem.
    :param control: optional int or string. Default None. Index or Name of the control algorithm. If control = None all FriedmanPosHocTest considers all possible comparisons among algorithms.
    :param apv_procedure: optional string. Default None.
        Name of the procedure for computing adjusted p-values. If apv_procedure
        is None, adjusted p-value are not computed, else the values are computed
        according to the specified procedure:
        For 1 vs all comparisons.
            {'Bonferroni', 'Holm', 'Hochberg', 'Holland', 'Finner', 'Li'}
        For all vs all coparisons.
            {'Shaffer', 'Holm', 'Nemenyi'}

    :return z_values: Test statistic.
    :return p_values: The p-value according to the Studentized range distribution.
    """

    # Initial Checking
    if type(data) == pd.DataFrame:
        algorithms = data.columns
        data = data.values
    elif type(data) == np.ndarray:
        algorithms = np.array(['Alg%d' % alg for alg in range(data.shape[1])])

    if control is None:
        index = algorithms
    elif type(control) == int:
        index = [algorithms[control]]
    else:
        index = [control]

    if data.ndim == 2:
        n_samples, k = data.shape
    else:
        raise ValueError(
            'Initialization ERROR. Incorrect number of array dimensions.')
    if k < 2:
        raise ValueError(
            'Initialization Error. Incorrect number of dimensions for axis 1.')

    if control is not None:
        if type(control) == int and control >= data.shape[1]:
            raise ValueError('Initialization ERROR. control is out of bounds')
        if type(control) == str and control not in algorithms:
            raise ValueError(
                'Initialization ERROR. %s is not a column name of data' % control)

    # Compute the average value achieved by all algorithms in each problem
    problemmean = np.mean(data, axis=1)
    # Compute the difference between control an data
    diff = np.zeros((n_samples, k))
    for j in range(k):
        diff[:, j] = data[:, j] - problemmean

    alignedRanks = ranks(np.ravel(diff))
    alignedRanks = np.reshape(alignedRanks, newshape=(n_samples, k))

    # Average ranks
    avranks = np.mean(alignedRanks, axis=0)

    # Compute test statistics
    aux = 1.0 / np.sqrt(k * (n_samples + 1) / 6.0)
    if control is None:
        z = np.zeros((k, k))
        for i in range(k):
            for j in range(i + 1, k):
                z[i, j] = abs(avranks[i] - avranks[j]) * aux
        z += z.T
    else:
        if type(control) == str:
            control = int(np.where(algorithms == control)[0])
        z = np.zeros((1, k))
        for j in range(k):
            z[0, j] = abs(avranks[control] - avranks[j]) * aux

    # Compute associated p-value
    p_value = 2 * (1.0 - norm.cdf(z))

    pvalues_df = pd.DataFrame(data=p_value, index=index, columns=algorithms)
    zvalues_df = pd.DataFrame(data=z, index=index, columns=algorithms)

    if apv_procedure is None:
        return zvalues_df, pvalues_df
    else:
        if apv_procedure == 'Bonferroni':
            ap_vs_df = bonferroni_dunn(pvalues_df, control=control)
        elif apv_procedure == 'Holm':
            ap_vs_df = holm(pvalues_df, control=control)
        elif apv_procedure == 'Hochberg':
            ap_vs_df = hochberg(pvalues_df, control=control)
        elif apv_procedure == 'Holland':
            ap_vs_df = holland(pvalues_df, control=control)
        elif apv_procedure == 'Finner':
            ap_vs_df = finner(pvalues_df, control=control)
        elif apv_procedure == 'Li':
            ap_vs_df = li(pvalues_df, control=control)
        elif apv_procedure == 'Shaffer':
            ap_vs_df = shaffer(pvalues_df)
        elif apv_procedure == 'Nemenyi':
            ap_vs_df = nemenyi(pvalues_df)

        return zvalues_df, pvalues_df, ap_vs_df


def quade_ph_test(data, control=None, apv_procedure=None):
    """ Quade post-hoc test.

    :param data: An (n x 2) array or DataFrame contaning the results. In data, each column represents an algorithm and, and each row a problem.
    :param control: optional int or string. Default None. Index or Name of the control algorithm. If control = None all FriedmanPosHocTest considers all possible comparisons among algorithms.
    :param apv_procedure: optional string. Default None.
        Name of the procedure for computing adjusted p-values. If apv_procedure
        is None, adjusted p-value are not computed, else the values are computed
        according to the specified procedure:
        For 1 vs all comparisons.
            {'Bonferroni', 'Holm', 'Hochberg', 'Holland', 'Finner', 'Li'}
        For all vs all coparisons.
            {'Shaffer', 'Holm', 'Nemenyi'}

    :return z_values: Test statistic.
    :return p_values: The p-value according to the Studentized range distribution.
    """

    # Initial Checking
    if type(data) == pd.DataFrame:
        algorithms = data.columns
        data = data.values
    elif type(data) == np.ndarray:
        algorithms = np.array(['Alg%d' % alg for alg in range(data.shape[1])])

    if control is None:
        index = algorithms
    elif type(control) == int:
        index = [algorithms[control]]
    else:
        index = [control]

    if data.ndim == 2:
        n_samples, k = data.shape
    else:
        raise ValueError(
            'Initialization ERROR. Incorrect number of array dimensions.')
    if k < 2:
        raise ValueError(
            'Initialization Error. Incorrect number of dimensions for axis 1.')

    if control is not None:
        if type(control) == int and control >= data.shape[1]:
            raise ValueError('Initialization ERROR. control is out of bounds')
        if type(control) == str and control not in algorithms:
            raise ValueError(
                'Initialization ERROR. %s is not a column name of data' % control)

    # Compute ranks.
    datarank = ranks(data)
    # Compute the range of each problem
    problemRange = np.max(data, axis=1) - np.min(data, axis=1)
    # Compute problem rank
    problemRank = ranks(problemRange)

    # Compute average rakings
    W = np.zeros((n_samples, k))
    for i in range(n_samples):
        W[i, :] = problemRank[i] * datarank[i, :]
    avranks = 2 * np.sum(W, axis=0) / (n_samples * (n_samples + 1))
    # Compute test statistics
    aux = 1.0 / np.sqrt(k * (k + 1) * (2 * n_samples + 1) * (k - 1) /
                        (18.0 * n_samples * (n_samples + 1)))
    if control is None:
        z = np.zeros((k, k))
        for i in range(k):
            for j in range(i + 1, k):
                z[i, j] = abs(avranks[i] - avranks[j]) * aux
        z += z.T
    else:
        if type(control) == str:
            control = int(np.where(algorithms == control)[0])
        z = np.zeros((1, k))
        for j in range(k):
            z[0, j] = abs(avranks[control] - avranks[j]) * aux

    # Compute associated p-value
    p_value = 2 * (1.0 - norm.cdf(z))

    pvalues_df = pd.DataFrame(data=p_value, index=index, columns=algorithms)
    zvalues_df = pd.DataFrame(data=z, index=index, columns=algorithms)

    if apv_procedure is None:
        return zvalues_df, pvalues_df
    else:
        if apv_procedure == 'Bonferroni':
            ap_vs_df = bonferroni_dunn(pvalues_df, control=control)
        elif apv_procedure == 'Holm':
            ap_vs_df = holm(pvalues_df, control=control)
        elif apv_procedure == 'Hochberg':
            ap_vs_df = hochberg(pvalues_df, control=control)
        elif apv_procedure == 'Holland':
            ap_vs_df = holland(pvalues_df, control=control)
        elif apv_procedure == 'Finner':
            ap_vs_df = finner(pvalues_df, control=control)
        elif apv_procedure == 'Li':
            ap_vs_df = li(pvalues_df, control=control)
        elif apv_procedure == 'Shaffer':
            ap_vs_df = shaffer(pvalues_df)
        elif apv_procedure == 'Nemenyi':
            ap_vs_df = nemenyi(pvalues_df)

        return zvalues_df, pvalues_df, ap_vs_df
