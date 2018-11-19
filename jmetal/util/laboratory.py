import io
import logging
from concurrent.futures import ProcessPoolExecutor
from itertools import chain

from scipy import stats
import pandas as pd

LOGGER = logging.getLogger('jmetal')

"""
.. module:: laboratory
   :platform: Unix, Windows
   :synopsis: Run experiments. WIP!

.. moduleauthor:: Antonio Benítez-Hidalgo <antonio.b@uma.es>
"""


class Experiment:

    def __init__(self, base_directory: str, algorithm_list: list, problem_list: list, metric_list: list,
                 n_runs: int = 1, m_workers: int = 3):
        """ Run an experiment to evaluate algorithms and/or problems.

        :param base_directory: Directory to save partial outputs.
        :param algorithm_list: List of algorithms as Tuple(Algorithm, dic() with parameters).
        :param metric_list: List of metrics. Each metric should inherit from :py:class:`Metric` or, at least,
        contain a method `compute`.
        :param m_workers: Maximum number of workers for ProcessPoolExecutor.
        """
        self.base_dir = base_directory

        self.algorithm_list = algorithm_list
        self.problem_list = problem_list
        self.metric_list = metric_list
        self.experiment_list = list()

        self.n_runs = n_runs
        self.m_workers = m_workers

    def run(self) -> None:
        self.__setup_runs()
        futures = []

        with ProcessPoolExecutor(max_workers=self.m_workers) as executor:
            for label, algorithm, n_run in self.experiment_list:
                # algorithm.observable.register(observer=WriteFrontToFileObserver(output_directory=self.base_dir + label))
                futures.append(executor.submit(algorithm.run()))

        print(futures)

    def compute_metrics(self) -> pd.DataFrame:
        runs = list(range(0, self.n_runs)) * len(self.metric_list) * len(self.problem_list)
        metrics = [[metric.get_name()] * self.n_runs for metric in self.metric_list] * len(self.problem_list)
        problems = [[problem.get_name()] * self.n_runs * len(self.metric_list) for problem in self.problem_list]

        arrays = [list(chain(*problems)), list(chain(*metrics)), runs]
        index = pd.MultiIndex.from_arrays(arrays, names=['problem', 'metric', 'run'])
        df = pd.DataFrame(data=None, columns=set([c['label'] for c in self.algorithm_list]), index=index, dtype=float)

        for label, algorithm, n_run in self.experiment_list:
            for metric in self.metric_list:
                r = metric.compute(algorithm.get_result())
                df.loc[(algorithm.problem.get_name(), metric.get_name(), n_run), label] = r

        return df

    def compute_statistical_analysis(self, data: pd.DataFrame):
        """ Compute the mean and standard deviation, median and interquartile range.

        :param data_list: List of data sets.
        """
        pass

    def compute_statiscal_analysis_2(self, data_list: list):
        """ The application scheme listed here is as described in

        * G. Luque, E. Alba, Parallel Genetic Algorithms, Springer-Verlag, ISBN 978-3-642-22084-5, 2011

        :param data_list: List of data sets.
        """
        if len(data_list) < 2:
            raise Exception('Data sets number must be equal or greater than two')

        dt = pd.DataFrame(data=None,
                          columns=[algorithm[0] for algorithm in self.algorithm_list],
                          index=[problem.get_name() for problem in self.problem_list])

        normality_test = True

        for data in data_list:
            statistic, pvalue = stats.kstest(data, 'norm')

            if pvalue > 0.05:
                normality_test = False
                break

        if not normality_test:
            # non-normal variables (median comparison, non-parametric tests)
            if len(data_list) == 2:
                statistic, pvalue = stats.wilcoxon(data_list[0], data_list[1])
            else:
                statistic, pvalue = stats.kruskal(*data_list)
        else:
            # normal variables (mean comparison, parametric tests)
            if len(data_list) == 2:
                pass
            else:
                pass

    def __setup_runs(self):
        """ Configure the algorithm list, by making a triple of (label, algorithm, n_run). """
        for n_run in range(self.n_runs):
            for configuration in self.algorithm_list:
                self.experiment_list.append((configuration['label'], configuration['algorithm'], n_run))

    def convert_to_latex(self, df: pd.DataFrame, caption: str='Experiment', label: str='tab:exp', alignment: str='c'):
        """ Convert a pandas dataframe to a LaTeX tabular. Prints labels in bold, does not use math mode. """
        num_columns, num_rows = df.shape[1], df.shape[0]
        output = io.StringIO()

        col_format = '{}|{}'.format(alignment, alignment * num_columns)
        column_labels = ['\\textbf{{{0}}}'.format(label.replace('_', '\\_')) for label in df.columns]

        # Write header
        output.write('\\begin{table}\n')
        output.write('\\caption{{{}}}\n'.format(caption))
        output.write('\\label{{{}}}\n'.format(label))
        output.write('\\centering\n')
        output.write('\\begin{scriptsize}\n')
        output.write('\\begin{tabular}{%s}\n' % col_format)
        output.write('\\hline\n')
        output.write('& {} \\\\\\hline\n'.format(' & '.join(column_labels)))

        # Write data lines
        for i in range(num_rows):
            output.write('\\textbf{{{0}}} & ${1}$ \\\\\n'.format(
                df.index[i], '$ & $'.join([str(val) for val in df.ix[i]]))
            )

        # Write footer
        output.write('\\hline\n')
        output.write('\\end{tabular}\n')
        output.write('\\end{scriptsize}\n')
        output.write('\\end{table}')

        return output.getvalue()
