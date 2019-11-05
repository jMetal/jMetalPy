Experiments
========================

Running the experiment
-----------------------

This is an example of an experimental study based on solving three problems of the ZDT family with three
different algorithms: NSGA-II, GDE3 and SMPSO.

The hypervolume, generational distance and epsilon indicators are used for performance assessment.

.. code-block:: python

    from jmetal.algorithm.multiobjective.gde3 import GDE3
    from jmetal.algorithm.multiobjective.nsgaii import NSGAII
    from jmetal.algorithm.multiobjective.smpso import SMPSO
    from jmetal.core.quality_indicator import *
    from jmetal.lab.experiment import Experiment, Job, generate_summary_from_experiment
    from jmetal.operator import PolynomialMutation, SBXCrossover
    from jmetal.problem import ZDT1, ZDT2, ZDT3
    from jmetal.util.archive import CrowdingDistanceArchive
    from jmetal.util.termination_criterion import StoppingByEvaluations


    def configure_experiment(problems: dict, n_run: int):
        jobs = []
        max_evaluations = 25000

        for run in range(n_run):
            for problem_tag, problem in problems.items():
                jobs.append(
                    Job(
                        algorithm=NSGAII(
                            problem=problem,
                            population_size=100,
                            offspring_population_size=100,
                            mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables,
                                                        distribution_index=20),
                            crossover=SBXCrossover(probability=1.0, distribution_index=20),
                            termination_criterion=StoppingByEvaluations(max=max_evaluations)
                        ),
                        algorithm_tag='NSGAII',
                        problem_tag=problem_tag,
                        run=run,
                    )
                )
                jobs.append(
                    Job(
                        algorithm=GDE3(
                            problem=problem,
                            population_size=100,
                            cr=0.5,
                            f=0.5,
                            termination_criterion=StoppingByEvaluations(max=max_evaluations)
                        ),
                        algorithm_tag='GDE3',
                        problem_tag=problem_tag,
                        run=run,
                    )
                )
                jobs.append(
                    Job(
                        algorithm=SMPSO(
                            problem=problem,
                            swarm_size=100,
                            mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables,
                                                        distribution_index=20),
                            leaders=CrowdingDistanceArchive(100),
                            termination_criterion=StoppingByEvaluations(max=max_evaluations)
                        ),
                        algorithm_tag='SMPSO',
                        problem_tag=problem_tag,
                        run=run,
                    )
                )

        return jobs


    if __name__ == '__main__':
        # Configure the experiments
        jobs = configure_experiment(problems={'ZDT1': ZDT1(), 'ZDT2': ZDT2(), 'ZDT3': ZDT3()}, n_run=31)

        # Run the study
        output_directory = 'data'
        experiment = Experiment(output_dir=output_directory, jobs=jobs)
        experiment.run()

Summary file
-----------------------

The results of this experiment can be summarized to a CSV file as follows:

.. code-block:: python

    if __name__ == '__main__':
        # experiment = ...

        # Generate summary file
        generate_summary_from_experiment(
            input_dir=output_directory,
            reference_fronts='/home/user/jMetalPy/resources/reference_front',
            quality_indicators=[GenerationalDistance(), EpsilonIndicator(), HyperVolume([1.0, 1.0])]
        )

This file contains all the information of the quality indicator values, for each configuration and run.
The summary file is the input of all the statistical tests, so that they can be applied to **any valid file having the proper format**.

.. code-block:: console

    $ head QualityIndicatorSummary.csv
    Algorithm,Problem,IndicatorName,ExecutionId,IndicatorValue
    NSGAII,ZDT1,EP,0,0.015705992620067832
    NSGAII,ZDT1,EP,1,0.012832504015918067
    ...

API
---

.. automodule:: jmetal.lab.experiment
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: LOGGER, R, generate_boxplot, generate_latex_tables, compute_wilcoxon, compute_mean_indicator, __averages_to_latex, __wilcoxon_to_latex, check_minimization

