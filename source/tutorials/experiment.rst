Experiments
========================

This is an example of an experimental study based on solving two problems of the ZDT family with two versions of the same algorithm (NSGA-II).
The hypervolume indicator is used for performance assessment.

.. code-block:: python

   from jmetal.util.laboratory import Experiment, Job

   def configure_experiment(problems: list, n_run: int):
    jobs = []

    for run in range(n_run):
        for problem in problems:
            jobs.append(
                Job(
                    algorithm=NSGAII(
                        problem=problem,
                        population_size=100,
                        mating_pool_size=100,
                        offspring_size=100,
                        max_evaluations=5000,
                        mutation=NullMutation(),
                        crossover=SBX(probability=1.0, distribution_index=20),
                        selection=BinaryTournamentSelection(comparator=RankingAndCrowdingDistanceComparator())
                    ),
                    label='NSGAII with Null Mutation',
                    run=run,
                )
            )
            jobs.append(
                Job(
                    algorithm=NSGAII(
                        problem=problem,
                        population_size=100,
                        mating_pool_size=100,
                        offspring_size=100,
                        max_evaluations=5000,
                        mutation=Polynomial(probability=1.0 / problem.number_of_variables, distribution_index=20),
                        crossover=SBX(probability=1.0, distribution_index=20),
                        selection=BinaryTournamentSelection(comparator=RankingAndCrowdingDistanceComparator())
                    ),
                    label='NSGAII with Polynomial Mutation',
                    run=run
                )
            )

    return jobs

   if __name__ == '__main__':
    jobs = configure_experiment(problems=[ZDT1(), ZDT2()], n_run=3)

    test = Experiment(
        base_directory='./experiment',
        jobs=jobs
    )
    test.run()

    metrics = [HyperVolume(reference_point=[1, 1]), ComputingTime()]
    data = test.compute_metrics(metrics)
    print(data)

.. table::

    +-------+-----------+----+-------------------------+-------------------------------+
    |       |           |    |NSGAII with Null Mutation|NSGAII with Polynomial Mutation|
    +-------+-----------+----+-------------------------+-------------------------------+
    |Problem|Metric     |Run |                         |                               |
    +=======+===========+====+=========================+===============================+
    |ZDT1   |Hypervolume|0   |0.315708                 |0.516258                       |
    +-------+-----------+----+-------------------------+-------------------------------+
    |       |           |1   |0.323271                 |0.491973                       |
    +-------+-----------+----+-------------------------+-------------------------------+
    |       |           |2   |0.414953                 |0.507568                       |
    +-------+-----------+----+-------------------------+-------------------------------+
    |ZDT2   |Hypervolume|0   |0.293225                 |0.504027                       |
    +-------+-----------+----+-------------------------+-------------------------------+
    |       |           |1   |0.280499                 |0.417048                       |
    +-------+-----------+----+-------------------------+-------------------------------+
    |       |           |2   |0.357358                 |0.489576                       |
    +-------+-----------+----+-------------------------+-------------------------------+

The return value of :code:`compute_metrics()` is a pandas DataFrame, so we can use all the methods that pandas makes available for us:

.. code-block:: python

   mean_results = data.groupby(['problem', 'metric']).mean()
   median_results = data.groupby(['problem', 'metric']).median()
   only_one_metric = data.xs('Hypervolume', level='metric')
   min_values = data.groupby(['problem', 'metric']).min()
   max_values = data.groupby(['problem', 'metric']).max()
