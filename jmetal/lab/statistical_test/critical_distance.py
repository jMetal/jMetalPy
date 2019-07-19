import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.stats.libqsturng import qsturng

from jmetal.lab.statistical_test.functions import ranks


def NemenyiCD(alpha: float, num_alg, num_dataset):
    """ Computes Nemenyi's critical difference:
    * CD = q_alpha * sqrt(num_alg*(num_alg + 1)/(6*num_prob))
    where q_alpha is the critical value, of the Studentized range statistic divided by sqrt(2).
    :param alpha: {0.1, 0.999}. Significance level.
    :param num_alg: number of tested algorithms.
    :param num_dataset: Number of problems/datasets where the algorithms have been tested.
    """

    # get critical value
    q_alpha = qsturng(p=1 - alpha, r=num_alg, v=num_alg * (num_dataset - 1)) / np.sqrt(2)

    # compute the critical difference
    cd = q_alpha * np.sqrt(num_alg * (num_alg + 1) / (6.0 * num_dataset))

    return cd


def CDplot(results, alpha: float = 0.05, higher_is_better: bool=False, alg_names: list = None, output_filename: str = 'cdplot.eps'):
    """ CDgraph plots the critical difference graph show in Janez Demsar's 2006 work:
    * Statistical Comparisons of Classifiers over Multiple Data Sets.
    :param results: A 2-D array containing results from each algorithm. Each row of 'results' represents an algorithm, and each column a dataset.
    :param alpha: {0.1, 0.999}. Significance level for the critical difference.
    :param alg_names: Names of the tested algorithms.
    """

    def _join_alg(avranks, num_alg, cd):
        """
        join_alg returns the set of non significant methods
        """

        # get all pairs
        sets = (-1) * np.ones((num_alg, 2))
        for i in range(num_alg):
            elements = np.where(np.logical_and(
                avranks - avranks[i] > 0, avranks - avranks[i] < cd))[0]
            if elements.size > 0:
                sets[i, :] = [avranks[i], avranks[elements[-1]]]
        sets = np.delete(sets, np.where(sets[:, 0] < 0)[0], axis=0)

        # group pairs
        group = sets[0, :]
        for i in range(1, sets.shape[0]):
            if sets[i - 1, 1] < sets[i, 1]:
                group = np.vstack((group, sets[i, :]))

        return group

    # Initial Checking
    if type(results) == pd.DataFrame:
        alg_names = results.index
        results = results.values
    elif type(results) == np.ndarray and alg_names is None:
        alg_names = np.array(
            ['Alg%d' % alg for alg in range(results.shape[1])])

    if results.ndim == 2:
        num_alg, num_dataset = results.shape
    else:
        raise ValueError(
            'Initialization ERROR: In CDplot(...) results must be 2-D array')

    # Get the critical difference
    cd = NemenyiCD(alpha, num_alg, num_dataset)

    # Compute ranks. (ranks[i][j] rank of the i-th algorithm on the j-th problem.)

    rranks = ranks(results.T, descending=higher_is_better)

    # Compute for each algorithm the ranking averages.
    avranks = np.transpose(np.mean(rranks, axis=0))
    indices = np.argsort(avranks).astype(np.uint8)
    avranks = avranks[indices]

    # Split algorithms.
    spoint = np.round(num_alg / 2.0).astype(np.uint8)
    leftalg = avranks[:spoint]
    rightalg = avranks[spoint:]
    rows = np.ceil(num_alg / 2.0).astype(np.uint8)

    # Figure settings.
    highest = np.ceil(np.max(avranks)).astype(np.uint8)  # highest shown rank
    lowest = np.floor(np.min(avranks)).astype(np.uint8)  # lowest shown rank
    width = 6  # default figure width (in inches)
    height = (0.575 * (rows + 1))  # figure height

    """
                        FIGURE
      (1,0)
        +-----+---------------------------+-------+
        |     |                           |       |
        |     |                           |       |
        |     |                           |       |
        +-----+---------------------------+-------+ stop
        |     |                           |       |
        |     |                           |       |
        |     |                           |       |
        |     |                           |       |
        |     |                           |       |
        |     |                           |       |
        +-----+---------------------------+-------+ sbottom
        |     |                           |       |
        +-----+---------------------------+-------+
            sleft                       sright     (0,1)
    """

    stop, sbottom, sleft, sright = 0.65, 0.1, 0.15, 0.85

    # main horizontal axis length
    lline = sright - sleft

    # Initialize figure
    fig = plt.figure(figsize=(width, height), facecolor='white')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()

    # Main horizontal axis
    ax.hlines(stop, sleft, sright, color='black', linewidth=0.7)
    for xi in range(highest - lowest + 1):
        # Plot mayor ticks
        ax.vlines(x=sleft + (lline * xi) / (highest - lowest),
                  ymin=stop, ymax=stop + 0.05, color='black', linewidth=0.7)
        # Mayor ticks labels
        ax.text(x=sleft + (lline * xi) / (highest - lowest),
                y=stop + 0.06,
                s=str(lowest + xi), ha='center', va='bottom')
        # Minor ticks
        if xi < highest - lowest:
            ax.vlines(x=sleft + (lline * (xi + 0.5)) / (highest - lowest),
                      ymin=stop, ymax=stop + 0.025, color='black', linewidth=0.7)

    # Plot lines/names for left models
    vspace = 0.5 * (stop - sbottom) / (spoint + 1)
    for i in range(spoint):
        ax.vlines(x=sleft + (lline * (leftalg[i] - lowest)) / (highest - lowest),
                  ymin=sbottom + (spoint - 1 - i) * vspace, ymax=stop, color='black', linewidth=0.7)
        ax.hlines(y=sbottom + (spoint - 1 - i) * vspace, xmin=sleft,
                  xmax=sleft +
                       (lline * (leftalg[i] - lowest)) / (highest - lowest),
                  color='black', linewidth=0.7)
        ax.text(x=sleft - 0.01, y=sbottom + (spoint - 1 - i) * vspace,
                s=alg_names[indices][i], ha='right', va='center')

    # Plot lines/names for right models
    vspace = 0.5 * (stop - sbottom) / (num_alg - spoint + 1)
    for i in range(num_alg - spoint):
        ax.vlines(x=sleft + (lline * (rightalg[i] - lowest)) / (highest - lowest),
                  ymin=sbottom + i * vspace, ymax=stop, color='black', linewidth=0.7)
        ax.hlines(y=sbottom + i * vspace,
                  xmin=sleft +
                       (lline * (rightalg[i] - lowest)) / (highest - lowest),
                  xmax=sright, color='black', linewidth=0.7)
        ax.text(x=sright + 0.01, y=sbottom + i * vspace,
                s=alg_names[indices][spoint + i], ha='left', va='center')

    # Plot critical difference rule
    if sleft + (cd * lline) / (highest - lowest) <= sright:
        ax.hlines(y=stop + 0.2, xmin=sleft, xmax=sleft +
                                                 (cd * lline) / (highest - lowest), linewidth=1.5)
        ax.text(x=sleft + 0.5 * (cd * lline) /
                  (highest - lowest), y=stop + 0.21, s='CD=%.3f' % cd, ha='center', va='bottom')
    else:
        ax.text(x=(sleft + sright) / 2, y=stop + 0.2, s='CD=%.3f' %
                                                        cd, ha='center', va='bottom')

    # Get pair of non-significant methods
    nonsig = _join_alg(avranks, num_alg, cd)
    if nonsig.ndim == 2:
        if nonsig.shape[0] == 2:
            left_lines = np.reshape(nonsig[0, :], (1, 2))
            right_lines = np.reshape(nonsig[1, :], (1, 2))
        else:
            left_lines = nonsig[:np.round(
                nonsig.shape[0] / 2.0).astype(np.uint8), :]
            right_lines = nonsig[np.round(
                nonsig.shape[0] / 2.0).astype(np.uint8):, :]
    else:
        left_lines = np.reshape(nonsig, (1, nonsig.shape[0]))

    # plot from the left
    vspace = 0.5 * (stop - sbottom) / (left_lines.shape[0] + 1)
    for i in range(left_lines.shape[0]):
        ax.hlines(y=stop - (i + 1) * vspace,
                  xmin=sleft + lline * (left_lines[i, 0] -
                                        lowest - 0.025) / (highest - lowest),
                  xmax=sleft + lline * (left_lines[i, 1] - lowest + 0.025) / (highest - lowest), linewidth=2)

    # plot from the rigth
    if nonsig.ndim == 2:
        vspace = 0.5 * (stop - sbottom) / (left_lines.shape[0])
        for i in range(right_lines.shape[0]):
            ax.hlines(y=stop - (i + 1) * vspace,
                      xmin=sleft + lline * (right_lines[i, 0] -
                                            lowest - 0.025) / (highest - lowest),
                      xmax=sleft + lline * (right_lines[i, 1] - lowest + 0.025) / (highest - lowest), linewidth=2)

    plt.savefig(output_filename, bbox_inches='tight')
    plt.show()
