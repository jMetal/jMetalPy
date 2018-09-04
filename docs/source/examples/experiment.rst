Experiments
========================

This is an example of an experimental study based on solving two problems of the ZDT family with two versions of the same algorithm (NSGAII).
The hypervolume indicator is used for performance assessment.

.. code-block:: python

   # Configure experiment
   problem_list = [ZDT1(), ZDT2()]
   algorithm_list = []

   for problem in problem_list:
      algorithm_list.append(
         ('NSGAII_A',
          NSGAII(
             problem=problem,
             population_size=100,
             max_evaluations=25000,
             mutation=NullMutation(),
             crossover=SBX(probability=1.0, distribution_index=20),
             selection=BinaryTournamentSelection(comparator=RankingAndCrowdingDistanceComparator())
          ))
      )
      algorithm_list.append(
         ('NSGAII_B',
          NSGAII(
             problem=problem,
             population_size=100,
             max_evaluations=25000,
             mutation=Polynomial(probability=1.0 / problem.number_of_variables, distribution_index=20),
             crossover=SBX(probability=1.0, distribution_index=20),
             selection=BinaryTournamentSelection(comparator=RankingAndCrowdingDistanceComparator())
          ))
      )

   study = Experiment(algorithm_list, n_runs=2)
   study.run()

   # Compute quality indicators
   metric_list = [HyperVolume(reference_point=[1, 1])]

   print(study.compute_metrics(metric_list))