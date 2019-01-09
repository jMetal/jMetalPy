import io
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from jmetal.core.algorithm import Algorithm
from jmetal.core.quality_indicator import QualityIndicator
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

    def __init__(self, output_dir: str, jobs: List[Job], m_workers: int = 6):
        """ Run an experiment to execute a list of jobs.

        :param output_dir: Base directory where each job will save its results.
        :param jobs: List of Jobs (from :py:mod:`jmetal.util.laboratory)`) to be executed.
        :param m_workers: Maximum number of workers to execute the Jobs in parallel.
        """
        self.jobs = jobs
        self.m_workers = m_workers
        self.output_dir = output_dir

    def run(self) -> None:
        with ProcessPoolExecutor(max_workers=self.m_workers) as executor:
            for job in self.jobs:
                output_path = os.path.join(self.output_dir, job.algorithm_tag, job.problem_tag)
                executor.submit(job.execute(output_path))


def generate_summary_from_experiment(input_dir: str, quality_indicators: List[QualityIndicator],
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


def compute_median_iqr_tables(filename: str):
    """ Compute the mean and IQR of each quality indicator.
    :param filename: Input summary file.
    """
    df = pd.read_csv(filename, skipinitialspace=True)

    if len(set(df.columns.tolist())) != 5:
        raise Exception('Wrong number of columns')

    median_iqr = pd.DataFrame()

    for algorithm_name, subset in df.groupby('Algorithm'):
        subset = subset.drop('Algorithm', axis=1)
        subset = subset.set_index(['Problem', 'IndicatorName', 'ExecutionId'])

        # Compute Median and Interquartile range
        median = subset.groupby(level=[0, 1]).median()
        iqr = subset.groupby(level=[0, 1]).quantile(0.75) - subset.groupby(level=[0, 1]).quantile(0.25)
        table = median.applymap('{:.2e}'.format) + '_{' + iqr.applymap('{:.2e}'.format) + '}'

        table = table.rename(columns={'IndicatorValue': algorithm_name})
        median_iqr = pd.concat([median_iqr, table], axis=1)

    for iqr_name, subset in median_iqr.groupby('IndicatorName'):
        subset.index = subset.index.droplevel(1)
        subset.to_csv('MedianIQR{}.csv'.format(iqr_name), sep='\t', encoding='utf-8')

        with open('MedianIQR{}.tex'.format(iqr_name), 'w') as latex:
            latex.write(__convert_table_to_latex(
                subset, caption='Median and Interquartile Range of the {} quality indicator'.format(iqr_name), label='')
            )


def compute_mean_indicator(filename: str, indicator_name: str):
    """ Compute the mean values of an indicator.
    :param filename:
    :param indicator_name: Quality indicator name.
    """
    df = pd.read_csv(filename, skipinitialspace=True)

    if len(set(df.columns.tolist())) != 5:
        raise Exception('Wrong number of columns')

    algorithms = pd.unique(df['Algorithm'])
    problems = pd.unique(df['Problem'])

    # We consider the quality indicator indicator_name
    data = df[df['IndicatorName'] == indicator_name]

    # Compute for each pair algorithm/problem the average of IndicatorValue
    average_values = np.zeros((problems.size, algorithms.size))
    j = 0
    for alg in algorithms:
        i = 0
        for pr in problems:
            average_values[i, j] = data['IndicatorValue'][np.logical_and(
                data['Algorithm'] == alg, data['Problem'] == pr)].mean()
            i += 1
        j += 1

    # Generate dataFrame from average values
    return pd.DataFrame(data=average_values, index=problems, columns=algorithms)


def __convert_table_to_latex(df: pd.DataFrame, caption: str, label: str, alignment: str = 'c'):
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
