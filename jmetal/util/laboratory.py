import io
import logging
from concurrent.futures import ProcessPoolExecutor
from typing import List

import pandas as pd
from scipy import stats

from jmetal.core.algorithm import Algorithm
from jmetal.component.quality_indicator import QualityIndicator

LOGGER = logging.getLogger('jmetal')

"""
.. module:: laboratory
   :platform: Unix, Windows
   :synopsis: Run experiments. WIP!

.. moduleauthor:: Antonio Benítez-Hidalgo <antonio.b@uma.es>
"""


class Job:

    def __init__(self, algorithm: Algorithm, problem_name: str, label: str, run: int):
        self.algorithm = algorithm
        self.problem_name = problem_name
        self.id_ = run
        self.label_ = label

        self.executed = False

    def run(self):
        self.algorithm.run()
        self.executed = True

    def evaluate(self, metric: QualityIndicator):
        if not self.executed:
            raise Exception('Algorithm must be run first')

        return metric.compute(self.algorithm)


class Experiment:

    def __init__(self, base_directory: str, jobs: list, m_workers: int = 3):
        """ Run an experiment to evaluate algorithms and/or problems.

        :param base_directory: Directory to save partial outputs.
        :param jobs: List of Jobs (from :py:mod:`jmetal.util.laboratory)`) to be executed.
        :param m_workers: Maximum number of workers to execute the Jobs in parallel.
        """
        self.base_dir = base_directory
        self.jobs = jobs
        self.m_workers = m_workers

    def run(self) -> None:
        with ProcessPoolExecutor(max_workers=self.m_workers) as executor:
            for job in self.jobs:
                executor.submit(job.run())

    def compute_metrics(self, metrics: list) -> pd.DataFrame:
        col_names = ['problem', 'metric', 'run']

        for job in self.jobs:
            if job.label_ not in col_names:
                col_names.append(job.label_)

        df = pd.DataFrame(columns=col_names)

        for job in self.jobs:
            for metric in metrics:
                value = job.evaluate(metric)
                new_data = pd.DataFrame(data=[[job.algorithm.problem.get_name(), metric.get_name(), job.id_, value]],
                                        columns=['problem', 'metric', 'run', job.label_])

                df = df.append(new_data)

        # Get rid of NaN values by grouping rows by columns
        df = df.groupby(['problem', 'metric', 'run']).mean()

        # Save to file
        df.to_csv(self.base_dir + '/metrics_df.csv', sep='\t', encoding='utf-8')

        return df

    def __compute_statistical_analysis(self, data_list: List[list]):
        """ The application scheme listed here is as described in

        * G. Luque, E. Alba, Parallel Genetic Algorithms, Springer-Verlag, ISBN 978-3-642-22084-5, 2011

        :param data_list: List of data sets.
        """
        if len(data_list) < 2:
            raise Exception('Data sets number must be equal or greater than two')

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

    @staticmethod
    def convert_to_latex(df: pd.DataFrame, caption: str = 'Experiment', label: str = 'tab:exp', alignment: str = 'c'):
        """ Convert a pandas DataFrame to a LaTeX tabular. Prints labels in bold, does not use math mode. """
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
