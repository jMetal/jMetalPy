import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def plot_posterior(sample, higher_is_better: bool = False, min_points_per_hexbin: int = 2, alg_names: list = None,
                   filename: str = 'posterior.eps'):
    """
    Plots the sample from posterior distribution of a Bayesian statistical test.
    Parameters:
    -----------
    data: An (n x 3) array or DataFrame contaning the probabilities.
    alg_names: array of strings. Default np.array(['Alg1', 'Alg2'])
        Names of the algorithms under evaluation
    Return:
    -------
    Figure
    """

    # Initial Checking
    if type(sample) == pd.DataFrame:
        sample = sample.values

    if sample.ndim == 2:
        nrow, ncol = sample.shape
        if ncol != 3:
            raise ValueError(
                'Initialization ERROR. Incorrect number of dimensions in axis 1.')
    else:
        raise ValueError(
            'Initialization ERROR. Incorrect number of dimensions for sample')

    def transform(p):
        lambda1, lambda2, lambda3 = p.T
        x = (0.1 * lambda1 + 0.5 * lambda2 + 0.9 * lambda3)
        y = (0.2 * lambda1 + 1.4 * lambda2 + 0.2 * lambda3) / np.sqrt(3)
        return np.vstack((x, y)).T

    # Initialize figure
    fig = plt.figure(figsize=(5, 5), facecolor='white')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()

    # plot text

    if not higher_is_better:
        if not alg_names:
            ax.text(x=0.5, y=1.4 / np.sqrt(3) + 0.005,
                    s='P(rope)', ha='center', va='bottom')
            ax.text(x=0.15, y=0.175 / np.sqrt(3) - 0.005,
                    s='P(alg1<alg2)', ha='right', va='top')
            ax.text(x=0.85, y=0.175 / np.sqrt(3) - 0.005,
                    s='P(alg1>alg2)', ha='left', va='top')
        else:
            ax.text(x=0.5, y=1.4 / np.sqrt(3) + 0.005,
                    s='P(rope)', ha='center', va='bottom')
            ax.text(x=0.15, y=0.175 / np.sqrt(3) - 0.005,
                    s='P(' + alg_names[0] + ')', ha='right', va='top')
            ax.text(x=0.85, y=0.175 / np.sqrt(3) - 0.005,
                    s='P(' + alg_names[1] + ')', ha='left', va='top')
    else:
        if not alg_names:
            ax.text(x=0.5, y=1.4 / np.sqrt(3) + 0.005,
                    s='P(rope)', ha='center', va='bottom')
            ax.text(x=0.15, y=0.175 / np.sqrt(3) - 0.005,
                    s='P(alg2<alg1)', ha='right', va='top')
            ax.text(x=0.85, y=0.175 / np.sqrt(3) - 0.005,
                    s='P(alg2>alg1)', ha='left', va='top')
        else:
            ax.text(x=0.5, y=1.4 / np.sqrt(3) + 0.005,
                    s='P(rope)', ha='center', va='bottom')
            ax.text(x=0.15, y=0.175 / np.sqrt(3) - 0.005,
                    s='P(' + alg_names[1] + ')', ha='right', va='top')
            ax.text(x=0.85, y=0.175 / np.sqrt(3) - 0.005,
                    s='P(' + alg_names[0] + ')', ha='left', va='top')

    # Conversion between barycentric and Cartesian coordinates
    sample2d = np.zeros((sample.shape[0], 2))
    for p in range(sample.shape[0]):
        sample2d[p, :] = transform(sample[p, :])

    # Plot projected points
    ax.hexbin(sample2d[:, 0], sample2d[:, 1], mincnt=min_points_per_hexbin, cmap=plt.cm.plasma)

    # Plot triangle

    ax.plot([0.095, 0.505], [0.2 / np.sqrt(3), 1.4 / np.sqrt(3)],
            linewidth=3.0, color='white')
    ax.plot([0.505, 0.905], [1.4 / np.sqrt(3), 0.2 / np.sqrt(3)],
            linewidth=3.0, color='white')
    ax.plot([0.09, 0.905], [0.2 / np.sqrt(3), 0.2 / np.sqrt(3)],
            linewidth=3.0, color='white')

    ax.plot([0.1, 0.5], [0.2 / np.sqrt(3), 1.4 / np.sqrt(3)],
            linewidth=3.0, color='gray')
    ax.plot([0.5, 0.9], [1.4 / np.sqrt(3), 0.2 / np.sqrt(3)],
            linewidth=3.0, color='gray')
    ax.plot([0.1, 0.9], [0.2 / np.sqrt(3), 0.2 / np.sqrt(3)],
            linewidth=3.0, color='gray')

    # plot division lines
    ax.plot([0.5, 0.5], [0.2 / np.sqrt(3), 0.6 / np.sqrt(3)],
            linewidth=3.0, color='gray')
    ax.plot([0.3, 0.5], [0.8 / np.sqrt(3), 0.6 / np.sqrt(3)],
            linewidth=3.0, color='gray')
    ax.plot([0.5, 0.7], [0.6 / np.sqrt(3), 0.8 / np.sqrt(3)],
            linewidth=3.0, color='gray')

    if filename:
        plt.savefig(filename, bbox_inches='tight')

    plt.show()
