import io
import logging
from concurrent.futures import ProcessPoolExecutor
from typing import List

import pandas as pd
from scipy import stats
from scipy.stats import friedmanchisquare

from jmetal.component.quality_indicator import QualityIndicator
from jmetal.core.algorithm import Algorithm
from jmetal.util.solution_list import print_function_values_to_file

LOGGER = logging.getLogger('jmetal')

"""
.. module:: laboratory
   :platform: Unix, Windows
   :synopsis: Run experiments. WIP!

.. moduleauthor:: Antonio Benítez-Hidalgo <antonio.b@uma.es>
"""


class Job:

    def __init__(self, algorithm: Algorithm, label: str, run: int):
        self.algorithm = algorithm
        self.id_ = run
        self.label_ = label

        self.executed = False

    def run(self):
        self.algorithm.run()
        self.executed = True

    def evaluate(self, metric: QualityIndicator):
        if not self.executed:
            self.run()

        if hasattr(metric, 'reference_front'):
            metric.reference_front = self.algorithm.problem.reference_front

        return metric.compute(self.algorithm.get_result())


class Experiment:

    def __init__(self, jobs: List[Job], m_workers: int = 3):
        """ Run an experiment to evaluate algorithms and/or problems.

        :param jobs: List of Jobs (from :py:mod:`jmetal.util.laboratory)`) to be executed.
        :param m_workers: Maximum number of workers to execute the Jobs in parallel.
        """
        self.jobs = jobs
        self.m_workers = m_workers

    def run(self) -> None:
        with ProcessPoolExecutor(max_workers=self.m_workers) as executor:
            for job in self.jobs:
                executor.submit(job.run())

    def compute_quality_indicator(self, qi: QualityIndicator) -> pd.DataFrame:
        pd.set_option('display.float_format', '{:.2e}'.format)
        df = pd.DataFrame()

        for job in self.jobs:
            new_data = pd.DataFrame({
                'problem': job.algorithm.problem.get_name(),
                'run': job.id_,
                job.label_: [job.evaluate(qi)]
            })
            df = df.append(new_data)

            # Save front to file
            file_name = 'data/{}/{}/FUN.{}.ps'.format(job.label_, job.algorithm.problem.get_name(), job.id_)
            print_function_values_to_file(job.algorithm.get_result(), file_name=file_name)

        # Get rid of NaN values by grouping rows by columns
        df = df.groupby(['problem', 'run']).mean()

        # Save to file
        LOGGER.debug('Saving output to experiment_df.csv')
        df.to_csv('data/experiment_df.csv', header=True, sep=',', encoding='utf-8')

        return df


def compute_statistical_analysis(df: pd.DataFrame):
    """ The application scheme listed here is as described in

    * G. Luque, E. Alba, Parallel Genetic Algorithms, Springer-Verlag, ISBN 978-3-642-22084-5, 2011

    :param df: Experiment data frame.
    """
    if len(df.columns) < 2:
        raise Exception('Data sets number must be equal or greater than two')

    statistic, pvalue = -1, -1
    result = pd.DataFrame()

    # we assume non-normal variables (median comparison, non-parametric tests)
    if len(df.columns) == 2:
        LOGGER.info('Running non-parametric test: Wilcoxon signed-rank test')
        statistic, pvalue = stats.wilcoxon(df[df.columns[0]], df[df.columns[1]])
    else:
        LOGGER.info('Running non-parametric test: Kruskal-Wallis test')
        for _, subset in df.groupby(level=0):
            statistic, pvalue = stats.kruskal(*subset.values.tolist())

            test = pd.DataFrame({
                'Kruskal-Wallis': '*' if pvalue < 0.05 else '-'
            }, index=[subset.index.values[0][0]], columns=['Kruskal-Wallis'])
            test.index.name = 'problem'

            result = result.append(test)

    return result


def convert_to_latex(df: pd.DataFrame, caption: str, label: str = 'tab:exp', alignment: str = 'c'):
    """ Convert a pandas DataFrame to a LaTeX tabular. Prints labels in bold, does not use math mode.
    """
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
