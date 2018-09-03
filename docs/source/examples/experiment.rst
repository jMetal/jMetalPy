Experiments
========================

This is an example of an experimental study based on solving two problems of the ZDT family with two versions of the same algorithm (NSGAII).
The hypervolume indicator is used for performance assessment.

.. code-block:: python

   # Configure the experiment
   algorithm = [
       (NSGAII, {'population_size': 100, 'max_evaluations': 25000, 'mutation': NullMutation(), 'crossover': SBX(1.0, 20),
                 'selection': BinaryTournamentSelection(RankingAndCrowdingDistanceComparator())}),
       (NSGAII, {'population_size': 100, 'max_evaluations': 25000, 'mutation': NullMutation(), 'crossover': SBX(0.3, 20),
                 'selection': BinaryTournamentSelection(RankingAndCrowdingDistanceComparator())})
   ]
   problem = [(ZDT1, {}), (ZDT2, {})]

   study = Experiment(algorithm, problem, n_runs=3)
   study.run()

   # Compute metrics
   metric = [HyperVolume(reference_point=[1, 1])]

   study.compute_metrics(metric)
