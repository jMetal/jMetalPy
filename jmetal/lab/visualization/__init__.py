try:
    from matplotlib.backends import _macosx
except ImportError:
    import matplotlib
    matplotlib.use('Qt5Agg')

from jmetal.lab.statistical_test.critical_distance import CDplot
from .interactive import InteractivePlot
from .plotting import Plot
from .posterior import plot_posterior
from .streaming import StreamingPlot

__all__ = [
    'Plot', 'InteractivePlot', 'StreamingPlot', 'CDplot', 'plot_posterior'
]
