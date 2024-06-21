import io
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import median
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, iqr, ks_2samp

from jmetal.core.algorithm import Algorithm
from jmetal.core.quality_indicator import QualityIndicator
from jmetal.logger import get_logger
from jmetal.util.solution import (
    print_function_values_to_file,
    print_variables_to_file,
    read_solutions,
)

logger = get_logger(__name__)

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

    def execute(self, output_path: str = ""):
        self.algorithm.run()

        if output_path:
            file_name = os.path.join(output_path, "FUN.{}.tsv".format(self.run_tag))
            print_function_values_to_file(self.algorithm.result(), filename=file_name)

            file_name = os.path.join(output_path, "VAR.{}.tsv".format(self.run_tag))
            print_variables_to_file(self.algorithm.result(), filename=file_name)

            file_name = os.path.join(output_path, "TIME.{}".format(self.run_tag))
            with open(file_name, "w+") as of:
                of.write(str(self.algorithm.total_computing_time))

    def get_algorithm_data(self):
        return self.algorithm.observable_data()


class Experiment:
    def __init__(self, output_dir: str, jobs: List[Job], m_workers: int = 6):
        """Run an experiment to execute a list of jobs.

        :param output_dir: Base directory where each job will save its results.
        :param jobs: List of Jobs (from :py:mod:`jmetal.util.laboratory)`) to be executed.
        :param m_workers: Maximum number of workers to execute the Jobs in parallel.
        """
        self.jobs = jobs
        self.m_workers = m_workers
        self.output_dir = output_dir
        self.job_data = []

    def run(self) -> None:
        with ProcessPoolExecutor(max_workers=self.m_workers) as executor:
            for job in self.jobs:
                output_path = os.path.join(self.output_dir, job.algorithm_tag, job.problem_tag)
                executor.submit(job.execute(output_path))
                self.job_data.append(job.get_algorithm_data())


def generate_summary_from_experiment(
        input_dir: str, quality_indicators: List[QualityIndicator], reference_fronts: str = ""
):
    """Compute a list of quality indicators. The input data directory *must* met the following structure (this is generated
    automatically by the Experiment class):

    * <base_dir>

      * algorithm_a

        * problem_a

          * FUN.0.tsv
          * FUN.1.tsv
          * VAR.0.tsv
          * VAR.1.tsv
          * ...

    :param input_dir: Directory where all the input data is found (function values and variables).
    :param reference_fronts: Directory where reference fronts are found.
    :param quality_indicators: List of quality indicators to compute.
    :return: None.
    """

    if not quality_indicators:
        quality_indicators = []

    with open("QualityIndicatorSummary.csv", "w+") as of:
        of.write("Algorithm,Problem,ExecutionId,IndicatorName,IndicatorValue\n")

    for dirname, _, filenames in os.walk(input_dir):
        for filename in filenames:
            try:
                # Linux filesystem
                algorithm, problem = dirname.split("/")[-2:]
            except ValueError:
                # Windows filesystem
                algorithm, problem = dirname.split("\\")[-2:]

            if "TIME" in filename:
                run_tag = [s for s in filename.split(".") if s.isdigit()].pop()

                with open(os.path.join(dirname, filename), "r") as content_file:
                    content = content_file.read()

                with open("QualityIndicatorSummary.csv", "a+") as of:
                    of.write(",".join([algorithm, problem, run_tag, "Time", str(content)]))
                    of.write("\n")

            if "FUN" in filename:
                solutions = read_solutions(os.path.join(dirname, filename))
                run_tag = [s for s in filename.split(".") if s.isdigit()].pop()
                for indicator in quality_indicators:
                    reference_front_file = os.path.join(reference_fronts, problem + ".pf")

                    # Add reference front if any
                    if hasattr(indicator, "reference_front"):
                        if Path(reference_front_file).is_file():
                            reference_front = []
                            with open(reference_front_file) as file:
                                for line in file:
                                    reference_front.append([float(x) for x in line.split()])

                            indicator.reference_front = reference_front
                        else:
                            logger.warning("Reference front not found at", reference_front_file)

                    result = indicator.compute([solutions[i].objectives for i in range(len(solutions))])

                    # Save quality indicator value to file
                    with open("QualityIndicatorSummary.csv", "a+") as of:
                        of.write(",".join([algorithm, problem, run_tag, indicator.get_short_name(), str(result)]))
                        of.write("\n")


def generate_boxplot(filename: str, output_dir: str = "boxplot"):
    """Generate boxplot diagrams.

    :param filename: Input filename (summary).
    :param output_dir: Output path.
    """
    df = pd.read_csv(filename, skipinitialspace=True)

    if len(set(df.columns.tolist())) != 5:
        raise Exception("Wrong number of columns")

    if Path(output_dir).is_dir():
        logger.warning("Directory {} exists. Removing contents.".format(output_dir))
        for file in os.listdir(output_dir):
            os.remove("{0}/{1}".format(output_dir, file))
    else:
        logger.warning("Directory {} does not exist. Creating it.".format(output_dir))
        Path(output_dir).mkdir(parents=True)

    algorithms = pd.unique(df["Algorithm"])
    problems = pd.unique(df["Problem"])
    indicators = pd.unique(df["IndicatorName"])

    # We consider the quality indicator indicator_name

    for indicator_name in indicators:
        data = df[df["IndicatorName"] == indicator_name]

        for pr in problems:
            data_to_plot = []

            for alg in algorithms:
                data_to_plot.append(
                    data["IndicatorValue"][np.logical_and(data["Algorithm"] == alg, data["Problem"] == pr)]
                )

            # Create a figure instance
            fig = plt.figure(1, figsize=(9, 6))
            plt.suptitle(pr, y=0.95, fontsize=18)

            ax = fig.add_subplot(111)
            ax.boxplot(data_to_plot)

            ax.set_xticklabels(algorithms)
            ax.tick_params(labelsize=20)

            plt.savefig(os.path.join(output_dir, "boxplot-{}-{}.png".format(pr, indicator_name)), bbox_inches="tight")
            plt.savefig(os.path.join(output_dir, "boxplot-{}-{}.eps".format(pr, indicator_name)), bbox_inches="tight")
            plt.close(fig)


def generate_latex_tables(filename: str, output_dir: str = "latex/statistical"):
    """Computes a number of statistical values (mean, median, standard deviation, interquartile range).

    :param filename: Input filename (summary).
    :param output_dir: Output path.
    """
    df = pd.read_csv(filename, skipinitialspace=True)

    if len(set(df.columns.tolist())) != 5:
        raise Exception("Wrong number of columns")

    if Path(output_dir).is_dir():
        logger.warning("Directory {} exists. Removing contents.".format(output_dir))
        for file in os.listdir(output_dir):
            os.remove("{0}/{1}".format(output_dir, file))
    else:
        logger.warning("Directory {} does not exist. Creating it.".format(output_dir))
        Path(output_dir).mkdir(parents=True)

    # Generate median & iqr tables
    median, iqr = pd.DataFrame(), pd.DataFrame()
    mean, std = pd.DataFrame(), pd.DataFrame()

    for algorithm_name, subset in df.groupby("Algorithm", sort=False):
        subset = subset.drop("Algorithm", axis=1)
        subset = subset.rename(columns={"IndicatorValue": algorithm_name})
        subset = subset.set_index(["Problem", "IndicatorName", "ExecutionId"])

        # Compute Median and Interquartile range
        median_ = subset.groupby(level=[0, 1]).median()
        median = pd.concat([median, median_], axis=1)

        iqr_ = subset.groupby(level=[0, 1]).quantile(0.75) - subset.groupby(level=[0, 1]).quantile(0.25)
        iqr = pd.concat([iqr, iqr_], axis=1)

        # Compute Mean and Standard deviation
        mean_ = subset.groupby(level=[0, 1]).mean()
        mean = pd.concat([mean, mean_], axis=1)

        std_ = subset.groupby(level=[0, 1]).std()
        std = pd.concat([std, std_], axis=1)

    # Generate mean & std tables
    for indicator_name, subset in std.groupby("IndicatorName", sort=False):
        subset = median.groupby("IndicatorName", sort=False).get_group(indicator_name)
        subset.index = subset.index.droplevel(1)
        subset.to_csv(os.path.join(output_dir, "Median-{}.csv".format(indicator_name)), sep="\t", encoding="utf-8")

        subset = iqr.groupby("IndicatorName", sort=False).get_group(indicator_name)
        subset.index = subset.index.droplevel(1)
        subset.to_csv(os.path.join(output_dir, "IQR-{}.csv".format(indicator_name)), sep="\t", encoding="utf-8")

        subset = mean.groupby("IndicatorName", sort=False).get_group(indicator_name)
        subset.index = subset.index.droplevel(1)
        subset.to_csv(os.path.join(output_dir, "Mean-{}.csv".format(indicator_name)), sep="\t", encoding="utf-8")

        subset = std.groupby("IndicatorName", sort=False).get_group(indicator_name)
        subset.index = subset.index.droplevel(1)
        subset.to_csv(os.path.join(output_dir, "Std-{}.csv".format(indicator_name)), sep="\t", encoding="utf-8")

    # Generate LaTeX tables
    for indicator_name in df.groupby("IndicatorName", sort=False).groups.keys():
        # Median & IQR
        md = median.groupby("IndicatorName", sort=False).get_group(indicator_name)
        md.index = md.index.droplevel(1)

        i = iqr.groupby("IndicatorName", sort=False).get_group(indicator_name)
        i.index = i.index.droplevel(1)

        with open(os.path.join(output_dir, "MedianIQR-{}.tex".format(indicator_name)), "w") as latex:
            latex.write(
                __averages_to_latex(
                    md,
                    i,
                    caption="Median and Interquartile Range of the {} quality indicator.".format(indicator_name),
                    minimization=check_minimization(indicator_name),
                    label="table:{}".format(indicator_name),
                )
            )

        # Mean & Std
        mn = mean.groupby("IndicatorName", sort=False).get_group(indicator_name)
        mn.index = mn.index.droplevel(1)

        s = std.groupby("IndicatorName", sort=False).get_group(indicator_name)
        s.index = s.index.droplevel(1)

        with open(os.path.join(output_dir, "MeanStd-{}.tex".format(indicator_name)), "w") as latex:
            latex.write(
                __averages_to_latex(
                    mn,
                    s,
                    caption="Mean and Standard Deviation of the {} quality indicator.".format(indicator_name),
                    minimization=check_minimization(indicator_name),
                    label="table:{}".format(indicator_name),
                )
            )


def compute_wilcoxon(filename: str, output_dir: str = "latex/wilcoxon"):
    """
    :param filename: Input filename (summary).
    :param output_dir: Output path.
    """
    df = pd.read_csv(filename, skipinitialspace=True)

    if len(set(df.columns.tolist())) != 5:
        raise Exception("Wrong number of columns")

    if Path(output_dir).is_dir():
        logger.warning("Directory {} exists. Removing contents.".format(output_dir))
        for file in os.listdir(output_dir):
            os.remove("{0}/{1}".format(output_dir, file))
    else:
        logger.warning("Directory {} does not exist. Creating it.".format(output_dir))
        Path(output_dir).mkdir(parents=True)

    algorithms = pd.unique(df["Algorithm"])
    problems = pd.unique(df["Problem"])
    indicators = pd.unique(df["IndicatorName"])

    table = pd.DataFrame(index=algorithms[0:-1], columns=algorithms[1:])

    for indicator_name in indicators:
        for i, row_algorithm in enumerate(algorithms[0:-1]):
            wilcoxon = []
            for j, col_algorithm in enumerate(algorithms[1:]):
                line = []

                if i <= j:
                    for problem in problems:
                        df1 = df[
                            (df["Algorithm"] == row_algorithm)
                            & (df["Problem"] == problem)
                            & (df["IndicatorName"] == indicator_name)
                            ]
                        df2 = df[
                            (df["Algorithm"] == col_algorithm)
                            & (df["Problem"] == problem)
                            & (df["IndicatorName"] == indicator_name)
                            ]

                        data1 = df1["IndicatorValue"]
                        data2 = df2["IndicatorValue"]

                        median1 = median(data1)
                        median2 = median(data2)

                        stat, p = mannwhitneyu(data1, data2)

                        if p <= 0.05:
                            if check_minimization(indicator_name):
                                if median1 <= median2:
                                    line.append("+")
                                else:
                                    line.append("o")
                            else:
                                if median1 >= median2:
                                    line.append("+")
                                else:
                                    line.append("o")
                        else:
                            line.append("-")
                    wilcoxon.append("".join(line))

            if len(wilcoxon) < len(algorithms):
                wilcoxon = [""] * (len(algorithms) - len(wilcoxon) - 1) + wilcoxon
            table.loc[row_algorithm] = wilcoxon

        table.to_csv(os.path.join(output_dir, "Wilcoxon-{}.csv".format(indicator_name)), sep="\t", encoding="utf-8")

        with open(os.path.join(output_dir, "Wilcoxon-{}.tex".format(indicator_name)), "w") as latex:
            latex.write(
                __wilcoxon_to_latex(
                    table,
                    caption="Wilcoxon values of the {} quality indicator ({}).".format(
                        indicator_name, ", ".join(problems)
                    ),
                    label="table:{}".format(indicator_name),
                )
            )


def compute_mean_indicator(filename: str, indicator_name: str):
    """Compute the mean values of an indicator.
    :param filename:
    :param indicator_name: Quality indicator name.
    """
    df = pd.read_csv(filename, skipinitialspace=True)

    if len(set(df.columns.tolist())) != 5:
        raise Exception("Wrong number of columns")

    algorithms = pd.unique(df["Algorithm"])
    problems = pd.unique(df["Problem"])

    # We consider the quality indicator indicator_name
    data = df[df["IndicatorName"] == indicator_name]

    # Compute for each pair algorithm/problem the average of IndicatorValue
    average_values = np.zeros((problems.size, algorithms.size))
    j = 0
    for alg in algorithms:
        i = 0
        for pr in problems:
            average_values[i, j] = data["IndicatorValue"][
                np.logical_and(data["Algorithm"] == alg, data["Problem"] == pr)
            ].mean()
            i += 1
        j += 1

    # Generate dataFrame from average values and order columns by name
    df = pd.DataFrame(data=average_values, index=problems, columns=algorithms)
    df = df.reindex(df.columns, axis=1)

    return df


def generate_median_and_wilcoxon_latex_tables(filename: str, output_dir: str = "latex/meansAndWilcoxon"):
    """Generate Latex tables including medians and IQRs. Additionally, the last algorithm is considered as the reference
        algorithm, and the cells include a symbol indicating whether the differences with the reference algorithm
        are significant or not according to the Wilcoxon rank sum test.

    :param filename: Input filename (summary).
    :param output_dir: Output path.
    """
    data = pd.read_csv(filename, skipinitialspace=True)

    if len(set(data.columns.tolist())) != 5:
        raise Exception("Wrong number of columns")

    if Path(output_dir).is_dir():
        logger.warning("Directory {} exists. Removing contents.".format(output_dir))
        for file in os.listdir(output_dir):
            os.remove("{0}/{1}".format(output_dir, file))
    else:
        logger.warning("Directory {} does not exist. Creating it.".format(output_dir))
        Path(output_dir).mkdir(parents=True)

    algorithms = pd.unique(data["Algorithm"])
    problems = pd.unique(data["Problem"])
    indicators = pd.unique(data["IndicatorName"])

    control_algorithm = algorithms[-1]

    # Compute medians and IQRs
    medians = data.groupby(["Algorithm", "Problem", "IndicatorName"])["IndicatorValue"].median()
    iqrs = data.groupby(["Algorithm", "Problem", "IndicatorName"])["IndicatorValue"].apply(lambda x: iqr(x))

    # Create data frame to store the Wilcoxon test results
    wilcoxon_data = pd.DataFrame(columns=["Indicator", "Algorithm", "Problem", "PValue", "Median", "TestResult"])

    for indicator in indicators:
        for algorithm in algorithms:
            for problem in problems:
                algorithm_data = data[(data["Problem"] == problem) & (data["Algorithm"] == algorithm) & (
                        data["IndicatorName"] == indicator)]
                ref_data = data[(data["Problem"] == problem) & (data["Algorithm"] == control_algorithm) & (
                        data["IndicatorName"] == indicator)]
                stat, p_value = mannwhitneyu(algorithm_data["IndicatorValue"], ref_data["IndicatorValue"])

                test_result = ""
                if p_value <= 0.05:
                    if check_minimization(indicator):
                        if medians[algorithm][problem][indicator] <= medians[control_algorithm][problem][indicator]:
                            test_result = '+'
                        else:
                            test_result = '-'
                    else:
                        if medians[algorithm][problem][indicator] >= medians[control_algorithm][problem][indicator]:
                            test_result = '+'
                        else:
                            test_result = '-'
                else:
                    test_result = '='

                new_row = {'Indicator': indicator, 'Algorithm': algorithm, "Problem": problem,
                           "PValue": p_value,
                           "Median": medians[algorithm][problem][indicator],
                           "IQR": iqrs[algorithm][problem][indicator],
                           "TestResult": test_result
                           }
                wilcoxon_data = wilcoxon_data._append(new_row, ignore_index=True)

    # Generate LaTeX tables
    caption = "Median and interquartile range (IQR) of the results of the {} quality indicator. " + \
              "Cells with dark and light gray background highlights, respectively, the best and second best indicator values. " + \
              "The algorithm in the last column is the reference " + \
              "algorithm, and the symbols $+$, $-$ and $\\approx$ indicate that the differences with the reference " + \
              "algorithm are significantly better, worse, or there is no difference according to the Wilcoxon rank " + \
              "sum test (confidence level: 95\%)."
    for indicator_name in indicators:
        with open(os.path.join(output_dir, "MedianIQRWilcoxon-{}.tex".format(indicator_name)), "w") as latex:
            latex.write(
                __median_wilcoxon_to_latex(
                    indicator_name,
                    wilcoxon_data,
                    caption=caption.format(indicator_name),
                    label="table:{}".format(indicator_name),
                )
            )


def generate_kolmogorov_smirnov_latex_tables(filename: str, output_dir: str = "latex/KolmogorovSmirnov"):
    """Generate Latex tables with the results of the Kolmogorov-Smirnov test. The last algorithm is considered as
        the reference algorithm, and the cells include a symbol with the p-value < 0.05.

    :param filename: Input filename (summary).
    :param output_dir: Output path.
    """
    data = pd.read_csv(filename, skipinitialspace=True)

    if len(set(data.columns.tolist())) != 5:
        raise Exception("Wrong number of columns")

    if Path(output_dir).is_dir():
        logger.warning("Directory {} exists. Removing contents.".format(output_dir))
        for file in os.listdir(output_dir):
            os.remove("{0}/{1}".format(output_dir, file))
    else:
        logger.warning("Directory {} does not exist. Creating it.".format(output_dir))
        Path(output_dir).mkdir(parents=True)

    algorithms = pd.unique(data["Algorithm"])
    problems = pd.unique(data["Problem"])
    indicators = pd.unique(data["IndicatorName"])

    control_algorithm = algorithms[-1]

    # Create data frame to store the Kolmogorov Smirnov test results
    test_data = pd.DataFrame(columns=["Indicator", "Algorithm", "Problem", "PValue", "TestResult"])

    for indicator in indicators:
        for algorithm in algorithms:
            for problem in problems:
                algorithm_data = data[(data["Problem"] == problem) & (data["Algorithm"] == algorithm) & (
                        data["IndicatorName"] == indicator)]
                ref_data = data[(data["Problem"] == problem) & (data["Algorithm"] == control_algorithm) & (
                        data["IndicatorName"] == indicator)]
                stat, p_value = ks_2samp(algorithm_data["IndicatorValue"], ref_data["IndicatorValue"])

                test_result = stat

                new_row = {'Indicator': indicator, 'Algorithm': algorithm, "Problem": problem,
                           "PValue": p_value,
                           "TestResult": test_result
                           }
                test_data = test_data._append(new_row, ignore_index=True)

    # Generate LaTeX tables
    caption = "Kolmogorov-Smirnov Test of the {} quality indicator. " \
              "The algorithm in the last column is the reference " + \
              "algorithm and each cell contain the p-value obtained when applying the test with the reference " \
              "algorithm. Cells with gray background highlight p-values less than 0.05 (i.e., the null hypothesis" \
              " -- the two distributions are identical -- is rejected)."
    for indicator_name in indicators:
        with open(os.path.join(output_dir, "KolmogorovSmirnov-{}.tex".format(indicator_name)), "w") as latex:
            latex.write(
                __kolmogorov_smirnov_to_latex(
                    indicator_name,
                    test_data,
                    caption=caption.format(indicator_name),
                    label="table:{}".format(indicator_name),
                )
            )


def __averages_to_latex(
        central_tendency: pd.DataFrame,
        dispersion: pd.DataFrame,
        caption: str,
        label: str,
        minimization=True,
        alignment: str = "c",
):
    """Convert a pandas DataFrame to a LaTeX tabular. Prints labels in bold and does use math mode.

    :param caption: LaTeX table caption.
    :param label: LaTeX table label.
    :param minimization: If indicator is minimization, highlight the best values of mean/median; else, the lowest.
    """
    num_columns, num_rows = central_tendency.shape[1], central_tendency.shape[0]
    output = io.StringIO()

    col_format = "{}|{}".format(alignment, alignment * num_columns)
    column_labels = ["\\textbf{{{0}}}".format(label.replace("_", "\\_")) for label in central_tendency.columns]

    # Write header
    output.write("\\documentclass{article}\n")

    output.write("\\usepackage[utf8]{inputenc}\n")
    output.write("\\usepackage{tabularx}\n")
    output.write("\\usepackage{colortbl}\n")
    output.write("\\usepackage[table*]{xcolor}\n")

    output.write("\\xdefinecolor{gray95}{gray}{0.65}\n")
    output.write("\\xdefinecolor{gray25}{gray}{0.8}\n")

    output.write("\\title{Median and IQR}\n")
    output.write("\\author{}\n")

    output.write("\\begin{document}\n")
    output.write("\\maketitle\n")

    output.write("\\section{Table}\n")

    output.write("\\begin{table}[!htp]\n")
    output.write("  \\caption{{{}}}\n".format(caption))
    output.write("  \\label{{{}}}\n".format(label))
    output.write("  \\centering\n")
    output.write("  \\begin{scriptsize}\n")
    output.write("  \\begin{tabular}{%s}\n" % col_format)
    output.write("      & {} \\\\\\hline\n".format(" & ".join(column_labels)))

    # Write data lines
    for i in range(num_rows):
        central_values = [v for v in central_tendency.iloc[i]]
        dispersion_values = [v for v in dispersion.iloc[i]]

        # Sort mean/median values (the lower the better if minimization)
        # Note that mean/median values could be the same: in that case, sort by Std/IQR (the lower the better)
        sorted_values = sorted(
            zip(central_values, dispersion_values, [i for i in range(len(central_values))]), key=lambda v: (v[0], -v[1])
        )

        if minimization:
            second_best, best = sorted_values[0][2], sorted_values[1][2]
        else:
            second_best, best = sorted_values[-1][2], sorted_values[-2][2]

        # Compose cell
        values = [
            "{:.2e}_{{{:.2e}}}".format(central_values[i], dispersion_values[i]) for i in range(len(central_values))
        ]

        # Highlight values
        values[best] = "\\cellcolor{gray25} " + values[best]
        values[second_best] = "\\cellcolor{gray95} " + values[second_best]

        output.write(
            "      \\textbf{{{0}}} & ${1}$ \\\\\n".format(
                central_tendency.index[i], " $ & $ ".join([str(val) for val in values])
            )
        )

    # Write footer
    output.write("  \\end{tabular}\n")
    output.write("  \\end{scriptsize}\n")
    output.write("\\end{table}\n")

    output.write("\\end{document}")

    return output.getvalue()


def __wilcoxon_to_latex(df: pd.DataFrame, caption: str, label: str, minimization=True, alignment: str = "c"):
    """Convert a pandas DataFrame to a LaTeX tabular. Prints labels in bold and does use math mode.

    :param df: Pandas dataframe.
    :param caption: LaTeX table caption.
    :param label: LaTeX table label.
    :param minimization: If indicator is minimization, highlight the best values of mean/median; else, the lowest.
    """
    num_columns, num_rows = df.shape[1], df.shape[0]
    output = io.StringIO()

    col_format = "{}|{}".format(alignment, alignment * num_columns)
    column_labels = ["\\textbf{{{0}}}".format(label.replace("_", "\\_")) for label in df.columns]

    # Write header
    output.write("\\documentclass{article}\n")

    output.write("\\usepackage[utf8]{inputenc}\n")
    output.write("\\usepackage{tabularx}\n")
    output.write("\\usepackage{amssymb}\n")
    output.write("\\usepackage{amsmath}\n")

    output.write("\\title{Wilcoxon - Mann-Whitney rank sum test}\n")
    output.write("\\author{}\n")

    output.write("\\begin{document}\n")
    output.write("\\maketitle\n")

    output.write("\\section{Table}\n")

    output.write("\\begin{table}[!htp]\n")
    output.write("  \\caption{{{}}}\n".format(caption))
    output.write("  \\label{{{}}}\n".format(label))
    output.write("  \\centering\n")
    output.write("  \\begin{scriptsize}\n")
    output.write("  \\begin{tabular}{%s}\n" % col_format)
    output.write("      & {} \\\\\\hline\n".format(" & ".join(column_labels)))

    symbolo = "\\triangledown\ "
    symbolplus = "\\blacktriangle\ "

    if not minimization:
        symbolo, symbolplus = symbolplus, symbolo

    # Write data lines
    for i in range(num_rows):
        values = [val.replace("-", "\\text{--}\ ").replace("o", symbolo).replace("+", symbolplus) for val in df.iloc[i]]
        output.write(
            "      \\textbf{{{0}}} & ${1}$ \\\\\n".format(df.index[i], " $ & $ ".join([str(val) for val in values]))
        )

    # Write footer
    output.write("  \\end{tabular}\n")
    output.write("  \\end{scriptsize}\n")
    output.write("\\end{table}\n")

    output.write("\\end{document}")

    return output.getvalue()


def __median_wilcoxon_to_latex(
        indicator_name: str,
        wilcoxon_data: pd.DataFrame,
        caption: str,
        label):
    indicator_data = wilcoxon_data[wilcoxon_data["Indicator"] == indicator_name]

    problems = pd.unique(indicator_data["Problem"])
    algorithms = pd.unique(indicator_data["Algorithm"])

    num_columns = len(algorithms)
    columns = algorithms

    alignment = "c"
    col_format = "{}|{}".format(alignment, alignment * num_columns)
    column_labels = ["\\textbf{{{0}}}".format(label.replace("_", "\\_")) for label in columns]

    output = io.StringIO()

    output.write("\\documentclass{article}\n")

    output.write("\\usepackage[utf8]{inputenc}\n")
    output.write("\\usepackage{tabularx}\n")
    output.write("\\usepackage{colortbl}\n")
    output.write("\\usepackage[table*]{xcolor}\n")

    output.write("\\xdefinecolor{gray95}{gray}{0.65}\n")
    output.write("\\xdefinecolor{gray25}{gray}{0.8}\n")

    output.write("\\title{Median and Wilcoxon}\n")
    output.write("\\author{}\n")

    output.write("\\begin{document}\n")
    output.write("\\maketitle\n")

    output.write("\\section{Table}\n")

    output.write("\\begin{table}[!htp]\n")
    output.write("  \\caption{{{}}}\n".format(caption))
    output.write("  \\label{{{}}}\n".format(label))
    output.write("  \\centering\n")
    output.write("  \\begin{tiny}\n")
    output.write("  \\begin{tabular}{%s}\n" % col_format)
    output.write("      & {} \\\\\\hline\n".format(" & ".join(column_labels)))

    # Counts the number of times that an algorithm performs better, worse or equal than the reference algorithm
    counters = {}
    for algorithm in algorithms:
        counters[algorithm] = [0, 0, 0]  # best, equal, worse

    for problem in problems:
        values = []

        for algorithm in algorithms:
            row = indicator_data[(indicator_data["Problem"] == problem) & (indicator_data["Algorithm"] == algorithm)]
            value = "{:.2e}({:.2e})".format(row["Median"].tolist()[0], row["IQR"].tolist()[0])

            # Include the symbol according to the Wilcoxon rank sum test with the reference algorithm
            if algorithm != algorithms[-1]:
                if row["TestResult"].tolist()[0] == "-":
                    value = "{{{}-}}".format(value)
                    counters[algorithm][2] = counters[algorithm][2] + 1
                elif row["TestResult"].tolist()[0] == "+":
                    value = "{{{}+}}".format(value)
                    counters[algorithm][0] = counters[algorithm][0] + 1
                else:
                    value = "{{{}\\approx}}".format(value)
                    counters[algorithm][1] = counters[algorithm][1] + 1
            values.append(value)

        # Find the best and second best values
        medians = indicator_data[(indicator_data["Problem"] == problem)]["Median"]
        iqrs = indicator_data[(indicator_data["Problem"] == problem)]["IQR"]
        pairs = list(zip(medians, iqrs))
        indexes = sorted(range(len(pairs)), key=lambda x: pairs[x])

        if check_minimization(indicator_name):
            best = indexes[0]
            second_best = indexes[1]
        else:
            best = indexes[-1]
            second_best = indexes[-2]

        values[best] = "\\cellcolor{gray95} " + values[best]
        values[second_best] = "\\cellcolor{gray25} " + values[second_best]

        output.write(
            "\\textbf{{{0}}} & ${1}$ \\\\\n".format(problem, " $ & $ ".join(
                [str(val).replace("e-", "e\makebox[0.1cm]{-}").replace("e+", "e\makebox[0.1cm]{+}") for val in values])
                                                    )
        )

    # Select all but the last counter
    counter_summary = []
    for algorithm in algorithms[:-1]:
        counter_summary.append(counters[algorithm])

    output.write("  \\hline\n")
    output.write(
        "\\textbf{{{0}}} & ${1}$ \\\\\n".format("$+/\\approx/-$", " $ & $ ".join(
            [str(val[0]) + "/" + str(val[1]) + "/" + str(val[2]) for val in counter_summary])))

    # Write footer
    output.write("  \\end{tabular}\n")
    output.write("  \\end{tiny}\n")
    output.write("\\end{table}\n")

    output.write("\\end{document}")

    return output.getvalue()


def __kolmogorov_smirnov_to_latex(indicator_name: str, test_data: pd.DataFrame, caption: str, label: str):
    indicator_data = test_data[test_data["Indicator"] == indicator_name]

    problems = pd.unique(indicator_data["Problem"])
    algorithms = pd.unique(indicator_data["Algorithm"])

    num_columns = len(algorithms)
    columns = algorithms

    alignment = "c"
    col_format = "{}|{}".format(alignment, alignment * num_columns)
    column_labels = ["\\textbf{{{0}}}".format(label.replace("_", "\\_")) for label in columns]

    output = io.StringIO()

    output.write("\\documentclass{article}\n")

    output.write("\\usepackage[utf8]{inputenc}\n")
    output.write("\\usepackage{tabularx}\n")
    output.write("\\usepackage{colortbl}\n")
    output.write("\\usepackage[table*]{xcolor}\n")

    output.write("\\xdefinecolor{gray95}{gray}{0.65}\n")
    output.write("\\xdefinecolor{gray25}{gray}{0.8}\n")

    output.write("\\title{Kolmogorov-Smirnov Test}\n")
    output.write("\\author{}\n")

    output.write("\\begin{document}\n")
    output.write("\\maketitle\n")

    output.write("\\section{Table}\n")

    output.write("\\begin{table}[!htp]\n")
    output.write("  \\caption{{{}}}\n".format(caption))
    output.write("  \\label{{{}}}\n".format(label))
    output.write("  \\centering\n")
    output.write("  \\begin{tiny}\n")
    output.write("  \\begin{tabular}{%s}\n" % col_format)
    output.write("      & {} \\\\\\hline\n".format(" & ".join(column_labels)))

    for problem in problems:
        values = []

        for algorithm in algorithms[:-1]:
            row = indicator_data[(indicator_data["Problem"] == problem) & (indicator_data["Algorithm"] == algorithm)]
            value = "{:.2e}".format(row["PValue"].tolist()[0])

            if (row["PValue"].tolist()[0] < 0.05):
                value = "\\cellcolor{gray25} " + value

            values.append(value)
        values.append("-")

        output.write(
            "\\textbf{{{0}}} & ${1}$ \\\\\n".format(problem, " $ & $ ".join(
                [str(val).replace("e-", "e\makebox[0.1cm]{-}").replace("e+", "e\makebox[0.1cm]{+}") for val in values])
                                                    )
        )

    output.write("  \\hline\n")

    # Write footer
    output.write("  \\end{tabular}\n")
    output.write("  \\end{tiny}\n")
    output.write("\\end{table}\n")

    output.write("\\end{document}")

    return output.getvalue()


def check_minimization(indicator) -> bool:
    if indicator == "HV":
        return False
    else:
        return True
