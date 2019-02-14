import numpy as np
import pandas as pd


def bonferroni_dunn(p_values, control):
    """
    Bonferroni-Dunn's procedure for the adjusted p-value computation.

    Parameters:
    -----------
    p_values: 2-D array or DataFrame containing the p-values obtained from a ranking test.
    control: int or string. Index or Name of the control algorithm.

    Returns:
    --------
    APVs: DataFrame containing the adjusted p-values.
    """

    # Initial Checking
    if type(p_values) == pd.DataFrame:
        algorithms = p_values.columns
        p_values = p_values.values
    elif type(p_values) == np.ndarray:
        algorithms = np.array(
            ['Alg%d' % alg for alg in range(p_values.shape[1])])

    if type(control) == str:
        control = int(np.where(algorithms == control)[0])
    if control is None:
        raise ValueError(
            'Initialization ERROR. Incorrect value for control.')

    k = p_values.shape[1]

    # sort p-values p(0) <= p(1) <= ... <= p(k-1)
    argsorted_pvals = np.argsort(p_values[0, :])

    APVs = np.zeros((k - 1, 1))
    comparison = []
    for i in range(k - 1):
        comparison.append(algorithms[control] +
                          ' vs ' + algorithms[argsorted_pvals[i]])
        APVs[i, 0] = np.min([(k - 1) * p_values[0, argsorted_pvals[i]], 1])
    return pd.DataFrame(data=APVs, index=comparison, columns=['Bonferroni'])


def holland(p_values, control):
    """
    Holland's procedure for the adjusted p-value computation.

    Parameters:
    -----------
    p_values: 2-D array or DataFrame containing the p-values obtained from a ranking test.
    control: int or string. Index or Name of the control algorithm.

    Returns:
    --------
    APVs: DataFrame containing the adjusted p-values.
    """

    # Initial Checking
    if type(p_values) == pd.DataFrame:
        algorithms = p_values.columns
        p_values = p_values.values
    elif type(p_values) == np.ndarray:
        algorithms = np.array(
            ['Alg%d' % alg for alg in range(p_values.shape[1])])

    if type(control) == str:
        control = int(np.where(algorithms == control)[0])
    if control is None:
        raise ValueError(
            'Initialization ERROR. Incorrect value for control.')

    # --------------------------------------------------------------------------
    # ------------------------------- Procedure --------------------------------
    # --------------------------------------------------------------------------
    k = p_values.shape[1]

    # sort p-values p(0) <= p(1) <= ... <= p(k-1)
    argsorted_pvals = np.argsort(p_values[0, :])

    APVs = np.zeros((k - 1, 1))
    comparison = []
    for i in range(k - 1):
        comparison.append(algorithms[control] +
                          ' vs ' + algorithms[argsorted_pvals[i]])
        aux = k - 1 - np.arange(i + 1)
        v = np.max(1 - (1 - p_values[0, argsorted_pvals[:(i + 1)]]) ** aux)
        APVs[i, 0] = np.min([v, 1])
    return pd.DataFrame(data=APVs, index=comparison, columns=['Holland'])


def finner(p_values, control):
    """
    Finner's procedure for the adjusted p-value computation.

    Parameters:
    -----------
    p_values: 2-D array or DataFrame containing the p-values obtained from a ranking test.
    control: int or string. Index or Name of the control algorithm.

    Returns:
    --------
    APVs: DataFrame containing the adjusted p-values.
    """

    # Initial Checking
    if type(p_values) == pd.DataFrame:
        algorithms = p_values.columns
        p_values = p_values.values
    elif type(p_values) == np.ndarray:
        algorithms = np.array(
            ['Alg%d' % alg for alg in range(p_values.shape[1])])

    if type(control) == str:
        control = int(np.where(algorithms == control)[0])
    if control is None:
        raise ValueError(
            'Initialization ERROR. Incorrect value for control.')

    k = p_values.shape[1]

    # sort p-values p(0) <= p(1) <= ... <= p(k-1)
    argsorted_pvals = np.argsort(p_values[0, :])

    APVs = np.zeros((k - 1, 1))
    comparison = []
    for i in range(k - 1):
        comparison.append(algorithms[control] +
                          ' vs ' + algorithms[argsorted_pvals[i]])
        aux = float(k - 1) / (np.arange(i + 1) + 1)
        v = np.max(1 - (1 - p_values[0, argsorted_pvals[:(i + 1)]]) ** aux)
        APVs[i, 0] = np.min([v, 1])
    return pd.DataFrame(data=APVs, index=comparison, columns=['Finner'])


def hochberg(p_values, control):
    """
    Hochberg's procedure for the adjusted p-value computation.

    Parameters:
    -----------
    p_values: 2-D array or DataFrame containing the p-values obtained from a ranking test.
    control: int or string. Index or Name of the control algorithm.

    Returns:
    --------
    APVs: DataFrame containing the adjusted p-values.
    """

    # Initial Checking
    if type(p_values) == pd.DataFrame:
        algorithms = p_values.columns
        p_values = p_values.values
    elif type(p_values) == np.ndarray:
        algorithms = np.array(
            ['Alg%d' % alg for alg in range(p_values.shape[1])])

    if type(control) == str:
        control = int(np.where(algorithms == control)[0])
    if control is None:
        raise ValueError(
            'Initialization ERROR. Incorrect value for control.')

    k = p_values.shape[1]

    # sort p-values p(0) <= p(1) <= ... <= p(k-1)
    argsorted_pvals = np.argsort(p_values[0, :])

    APVs = np.zeros((k - 1, 1))
    comparison = []
    for i in range(k - 1):
        comparison.append(algorithms[control] +
                          ' vs ' + algorithms[argsorted_pvals[i]])
        aux = np.arange(k, i, -1).astype(np.uint8)
        v = np.max(p_values[0, argsorted_pvals[aux - 1]] * (k - aux))
        APVs[i, 0] = np.min([v, 1])
    return pd.DataFrame(data=APVs, index=comparison, columns=['Hochberg'])


def li(p_values, control):
    """
    Li's procedure for the adjusted p-value computation.

    Parameters:
    -----------
    p_values: 2-D array or DataFrame containing the p-values obtained from a ranking test.
    control: optional int or string. Default None
        Index or Name of the control algorithm. If control is provided, control vs all
        comparisons are considered, else all vs all.

    Returns:
    --------
    APVs: DataFrame containing the adjusted p-values.
    """

    # Initial Checking
    if type(p_values) == pd.DataFrame:
        algorithms = p_values.columns
        p_values = p_values.values
    elif type(p_values) == np.ndarray:
        algorithms = np.array(
            ['Alg%d' % alg for alg in range(p_values.shape[1])])

    if type(control) == str:
        control = int(np.where(algorithms == control)[0])
    if control is None:
        raise ValueError(
            'Initialization ERROR. Incorrect value for control.')

    k = p_values.shape[1]

    # sort p-values p(0) <= p(1) <= ... <= p(k-1)
    argsorted_pvals = np.argsort(p_values[0, :])

    APVs = np.zeros((k - 1, 1))
    comparison = []
    for i in range(k - 1):
        comparison.append(algorithms[control] +
                          ' vs ' + algorithms[argsorted_pvals[i]])
        APVs[i, 0] = np.min([p_values[0, argsorted_pvals[-2]], p_values[0, argsorted_pvals[i]] / (
                p_values[0, argsorted_pvals[i]] + 1 - p_values[0, argsorted_pvals[-2]])])
    return pd.DataFrame(data=APVs, index=comparison, columns=['Li'])


def holm(p_values, control=None):
    """
    Holm's procedure for the adjusted p-value computation.

    Parameters:
    -----------
    p_values: 2-D array or DataFrame containing the p-values obtained from a ranking test.
    control: optional int or string. Default None
        Index or Name of the control algorithm. If control is provided, control vs all
        comparisons are considered, else all vs all.

    Returns:
    --------
    APVs: DataFrame containing the adjusted p-values.
    """

    # Initial Checking
    if type(p_values) == pd.DataFrame:
        algorithms = p_values.columns
        p_values = p_values.values
    elif type(p_values) == np.ndarray:
        algorithms = np.array(
            ['Alg%d' % alg for alg in range(p_values.shape[1])])

    if type(control) == str:
        control = int(np.where(algorithms == control)[0])

    if type(control) == int:
        k = p_values.shape[1]

        # sort p-values p(0) <= p(1) <= ... <= p(k-1)
        argsorted_pvals = np.argsort(p_values[0, :])

        APVs = np.zeros((k - 1, 1))
        comparison = []
        for i in range(k - 1):
            aux = k - 1 - np.arange(i + 1)
            comparison.append(
                algorithms[control] + ' vs ' + algorithms[argsorted_pvals[i]])
            v = np.max(aux * p_values[0, argsorted_pvals[:(i + 1)]])
            APVs[i, 0] = np.min([v, 1])

    elif control is None:
        k = p_values.shape[1]
        m = int((k * (k - 1)) / 2.0)

        # sort p-values p(0) <= p(1) <= ... <= p(m-1)
        pairs_index = np.triu_indices(k, 1)
        pairs_pvals = p_values[pairs_index]
        pairs_sorted = np.argsort(pairs_pvals)

        APVs = np.zeros((m, 1))
        aux = pairs_pvals[pairs_sorted] * (m - np.arange(m))
        comparison = []
        for i in range(m):
            row = pairs_index[0][pairs_sorted[i]]
            col = pairs_index[1][pairs_sorted[i]]
            comparison.append(algorithms[row] + ' vs ' + algorithms[col])
            v = np.max(aux[:i + 1])
            APVs[i, 0] = np.min([v, 1])
    return pd.DataFrame(data=APVs, index=comparison, columns=['Holm'])


def shaffer(p_values):
    """
    Shaffer's procedure for adjusted p_value ccmputation.

    Parameters:
    -----------
    data: 2-D array or DataFrame containing the p-values.

    Returns:
    --------
    APVs: DataFrame containing the adjusted p-values.
    """

    def S(k):
        """
        Computes the set of possible numbers of true hoypotheses.

        Parameters:
        -----------
        k: int
            number of algorithms being compared.

        Returns
        ----------
        TrueSet : array-like
            Set of true hypotheses.
        """

        from scipy.special import binom as binomial

        TrueHset = [0]
        if k > 1:
            for j in np.arange(k, 0, -1, dtype=int):
                TrueHset = list(set(TrueHset) | set(
                    [binomial(j, 2) + x for x in S(k - j)]))
        return TrueHset

    # Initial Checking
    if type(p_values) == pd.DataFrame:
        algorithms = p_values.columns
        p_values = p_values.values
    elif type(p_values) == np.ndarray:
        algorithms = np.array(
            ['Alg%d' % alg for alg in range(p_values.shape[1])])

    if p_values.ndim != 2:
        raise ValueError(
            'Initialization ERROR. Incorrect number of array dimensions.')
    elif p_values.shape[0] != p_values.shape[1]:
        raise ValueError(
            'Initialization ERROR. Incorrect number of array dimensions.')

    # define parameters
    k = p_values.shape[0]
    m = int(k * (k - 1) / 2.0)
    s = np.array(S(k)[1:])

    # sort p-values p(0) <= p(1) <= ... <= p(m-1)
    pairs_index = np.triu_indices(k, 1)
    pairs_pvals = p_values[pairs_index]
    pairs_sorted = np.argsort(pairs_pvals)

    # compute ti: max number of hypotheses that can be true given that any
    # (i-1) hypotheses are false.
    t = np.sort(-np.repeat(s[:-1], (s[1:] - s[:-1]).astype(np.uint8)))
    t = np.insert(-t, 0, s[-1])

    # Adjust p-values
    APVs = np.zeros((m, 1))
    aux = (pairs_pvals[pairs_sorted] * t)
    comparison = []
    for i in range(m):
        row = pairs_index[0][pairs_sorted[i]]
        col = pairs_index[1][pairs_sorted[i]]
        comparison.append(algorithms[row] + ' vs ' + algorithms[col])
        v = np.max(aux[:i + 1])
        APVs[i, 0] = np.min([v, 1])
    return pd.DataFrame(data=APVs, index=comparison, columns=['Shaffer'])


def nemenyi(p_values):
    """
    Nemenyi's procedure for adjusted p_value computation.

    Parameters:
    -----------
    data: 2-D array or DataFrame containing the p-values.

    Returns:
    --------
    APVs: DataFrame containing the adjusted p-values.
    """

    # Initial Checking
    if type(p_values) == pd.DataFrame:
        algorithms = p_values.columns
        p_values = p_values.values
    elif type(p_values) == np.ndarray:
        algorithms = np.array(
            ['Alg%d' % alg for alg in range(p_values.shape[1])])

    if p_values.ndim != 2:
        raise ValueError(
            'Initialization ERROR. Incorrect number of array dimensions.')
    elif p_values.shape[0] != p_values.shape[1]:
        raise ValueError(
            'Initialization ERROR. Incorrect number of array dimensions.')

    # define parameters
    k = p_values.shape[0]
    m = int(k * (k - 1) / 2.0)

    # sort p-values p(0) <= p(1) <= ... <= p(m-1)
    pairs_index = np.triu_indices(k, 1)
    pairs_pvals = p_values[pairs_index]
    pairs_sorted = np.argsort(pairs_pvals)

    # Adjust p-values
    APVs = np.zeros((m, 1))
    comparison = []
    for i in range(m):
        row = pairs_index[0][pairs_sorted[i]]
        col = pairs_index[1][pairs_sorted[i]]
        comparison.append(algorithms[row] + ' vs ' + algorithms[col])
        APVs[i, 0] = np.min([pairs_pvals[pairs_sorted[i]] * m, 1])
    return pd.DataFrame(data=APVs, index=comparison, columns=['Nemenyi'])
