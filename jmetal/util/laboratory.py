import io
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List

import pandas as pd
from scipy import stats

from jmetal.component.quality_indicator import QualityIndicator
from jmetal.core.algorithm import Algorithm
from jmetal.util.solution_list import print_function_values_to_file, print_variables_to_file, read_solutions

LOGGER = logging.getLogger('jmetal')

"""
.. module:: laboratory
   :platform: Unix, Windows
   :synopsis: Run experiments. WIP!

.. moduleauthor:: Antonio Benítez-Hidalgo <antonio.b@uma.es>
"""


class Job:

    def __init__(self, algorithm: Algorithm, algorithm_tag: str, problem_tag: str, run: int):
        self.algorithm = algorithm
        self.algorithm_tag = algorithm_tag
        self.problem_tag = problem_tag
        self.run_tag = run

    def execute(self, output_path: str = ''):
        self.algorithm.run()

        if output_path:
            file_name = os.path.join(output_path, 'FUN.{}.tsv'.format(self.run_tag))
            print_function_values_to_file(self.algorithm.get_result(), file_name=file_name)

            file_name = os.path.join(output_path, 'VAR.{}.tsv'.format(self.run_tag))
            print_variables_to_file(self.algorithm.get_result(), file_name=file_name)


class Experiment:

    def __init__(self, base_dir: str, jobs: List[Job], m_workers: int = 6):
        """ Run an experiment to execute a list of jobs.

        :param base_dir: Base directory where each job will save its results.
        :param jobs: List of Jobs (from :py:mod:`jmetal.util.laboratory)`) to be executed.
        :param m_workers: Maximum number of workers to execute the Jobs in parallel.
        """
        self.jobs = jobs
        self.m_workers = m_workers
        self.base_dir = base_dir

    def run(self) -> None:
        # todo This doesn't seems to be working properly
        with ProcessPoolExecutor(max_workers=self.m_workers) as executor:
            for job in self.jobs:
                output_path = os.path.join(self.base_dir, job.algorithm_tag, job.problem_tag)
                executor.submit(job.execute(output_path))


def compute_quality_indicator(input_dir: str, quality_indicators: List[QualityIndicator],
                              reference_fronts: str = ''):
    """ Compute a list of quality indicators. The input data directory *must* met the following structure (this is generated
    automatically by the Experiment class):

    * <base_dir>

      * algorithm_a

        * problem_a

          * FUN.0.tsv
          * FUN.1.tsv
          * VAR.0.tsv
          * VAR.1.tsv
          * ...

        * problem_b

          * FUN.0.tsv
          * FUN.1.tsv
          * VAR.0.tsv
          * VAR.1.tsv
          * ...

      * algorithm_b

        * ...

    For each indicator a new file `QI.<name_of_the_indicator>` is created inside each problem folder, containing the values computed for each front.

    :param input_dir: Directory where all the input data is found (function values and variables).
    :param reference_fronts: Directory where reference fronts are found.
    :param quality_indicators: List of quality indicators to compute.
    :return: None.
    """

    with open(os.path.join(input_dir, 'QualityIndicatorSummary.csv'), 'w+') as of:
        of.write('Algorithm,Problem,ExecutionId,IndicatorName,IndicatorValue')

    for dirname, _, filenames in os.walk(input_dir):
        for filename in filenames:
            algorithm, problem = dirname.split('/')[-2:]

            if 'FUN' in filename:
                solutions = read_solutions(os.path.join(dirname, filename))

                for indicator in quality_indicators:
                    reference_front_file = os.path.join(reference_fronts, problem + '.pf')

                    # Add reference front if found
                    if hasattr(indicator, 'reference_front'):
                        if Path(reference_front_file).is_file():
                            indicator.reference_front = read_solutions(reference_front_file)
                        else:
                            LOGGER.warning('Reference front not found at', reference_front_file)

                    run_tag = [int(s) for s in filename.split('.') if s.isdigit()].pop()
                    result = indicator.compute(solutions)

                    # Save quality indicator value to file
                    # Note: We need to ensure that the result is inserted at the correct row inside the file
                    with open(os.path.join(input_dir, 'QI.Summary.csv'), 'a+') as of:
                        of.write(','.join([algorithm, problem, run_tag, indicator.get_name(), result]))

                    with open(os.path.join(dirname, 'QI.' + indicator.get_name()), 'a+') as of:
                        contents = of.readlines()
                        contents.insert(run_tag, str(result) + '\n')

                        of.seek(0)  # readlines consumes the iterator, so we need to start over
                        of.writelines(contents)


def create_tables_from_experiment(base_dir: str, filename: str):
    # pd.set_option('display.float_format', '{:.2e}'.format)
    df = pd.read_csv(os.path.join(base_dir, filename), skipinitialspace=True)

    if {'Problem', 'ExecutionId', 'IndicatorName', 'IndicatorValue'} == set(df.columns.tolist()):
        raise Exception('Wrong column names')

    median_iqr = pd.DataFrame()

    for algorithm_name, subset in df.groupby('Algorithm'):
        subset = subset.drop('Algorithm', axis=1)
        subset = subset.set_index(['Problem', 'IndicatorName', 'ExecutionId'])
        subset.to_csv(os.path.join(base_dir, 'QualityIndicator' + algorithm_name + '.csv'), sep='\t', encoding='utf-8')

        # Compute Median and Interquartile range
        median = subset.groupby(level=[0, 1]).median()
        iqr = subset.groupby(level=[0, 1]).quantile(0.75) - subset.groupby(level=[0, 1]).quantile(0.25)
        table = median.applymap('{:.2e}'.format) + '_{' + iqr.applymap('{:.2e}'.format) + '}'
        table = table.rename(columns={'IndicatorValue': algorithm_name})

        median_iqr = pd.concat([median_iqr, table], axis=1)

    median_iqr.to_csv(os.path.join(base_dir, 'MedianIQR.csv'), sep='\t', encoding='utf-8')

    for iqr_name, subset in median_iqr.groupby('IndicatorName'):
        subset.index = subset.index.droplevel(1)
        subset.to_csv(os.path.join(base_dir, 'MedianIQR{}.csv'.format(iqr_name)), sep='\t', encoding='utf-8')


def __compute_statistical_analysis(df: pd.DataFrame):
    """ The application scheme listed here is as described in

    * G. Luque, E. Alba, Parallel Genetic Algorithms, Springer-Verlag, ISBN 978-3-642-22084-5, 2011

    ..note: We assume non-normal variables (median comparison, non-parametric tests).

    :param df: Experiment data frame.
    """
    if len(df.columns) < 2:
        raise Exception('At least two algorithms are necessary to compare')

    result = pd.DataFrame()

    if len(df.columns) == 2:
        LOGGER.info('Running non-parametric test: Wilcoxon signed-rank test')
        for _, subset in df.groupby(level=0):
            statistic, pvalue = stats.wilcoxon(subset[subset.columns[0]], subset[subset.columns[1]])

            test = pd.DataFrame({
                'Wilcoxon': '*' if pvalue < 0.05 else '-'
            }, index=[subset.index.values[0][0]], columns=['Wilcoxon'])
            test.index.name = 'problem'

            result = result.append(test)
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


def convert_table_to_latex(df: pd.DataFrame, caption: str, label: str = 'tab:exp', alignment: str = 'c'):
    """ Convert a pandas DataFrame to a LaTeX tabular. Prints labels in bold and does use math mode.
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
