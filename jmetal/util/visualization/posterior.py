import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_posterior(sample):
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

    # Plot triangle
    ax.plot([0.1, 0.5], [0.2 / np.sqrt(3), 1.4 / np.sqrt(3)],
            linewidth=1.0, color='grey')
    ax.plot([0.5, 0.9], [1.4 / np.sqrt(3), 0.2 / np.sqrt(3)],
            linewidth=1.0, color='grey')
    ax.plot([0.1, 0.9], [0.2 / np.sqrt(3), 0.2 / np.sqrt(3)],
            linewidth=1.0, color='grey')

    # plot division lines
    ax.plot([0.5, 0.5], [0.2 / np.sqrt(3), 0.6 / np.sqrt(3)],
            linewidth=1.0, color='grey')
    ax.plot([0.3, 0.5], [0.8 / np.sqrt(3), 0.6 / np.sqrt(3)],
            linewidth=1.0, color='grey')
    ax.plot([0.5, 0.7], [0.6 / np.sqrt(3), 0.8 / np.sqrt(3)],
            linewidth=1.0, color='grey')

    # plot text
    ax.text(x=0.5, y=1.4 / np.sqrt(3) + 0.005,
            s='rope', ha='center', va='bottom')
    ax.text(x=0.1, y=0.2 / np.sqrt(3) - 0.005,
            s='left', ha='right', va='top')
    ax.text(x=0.9, y=0.2 / np.sqrt(3) - 0.005,
            s='right', ha='left', va='top')

    # Conversion between barycentric and Cartesian coordinates
    sample2d = np.zeros((sample.shape[0], 2))
    for p in range(sample.shape[0]):
        sample2d[p, :] = transform(sample[p, :])

    # Plot projected points
    ax.hexbin(sample2d[:, 0], sample2d[:, 1], mincnt=1, cmap=plt.cm.plasma)

    return fig
