Examples of running EAs
========================

NSGA-II
------------------------------------

.. code-block:: python

    problem = ZDT1()
    problem.reference_front = read_solutions(filename='/resources/reference_front/{}.pf'.format(problem.get_name()))

    max_evaluations = 25000
    algorithm = NSGAII(
        problem=problem,
        population_size=100,
        offspring_population_size=100,
        mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20),
        crossover=SBXCrossover(probability=1.0, distribution_index=20),
        selection=BinaryTournamentSelection(comparator=RankingAndCrowdingDistanceComparator()),
        termination_criterion=StoppingByEvaluations(max=max_evaluations),
        dominance_comparator=DominanceComparator()
    )

    algorithm.observable.register(observer=ProgressBarObserver(max=max_evaluations))

    algorithm.run()
    front = algorithm.get_result()


MOEA/D
------------------------------------

.. code-block:: python

    problem = LZ09_F2()
    problem.reference_front = read_solutions(filename='/resources/reference_front/{}.pf'.format(problem.get_name()))

    max_evaluations = 150000

    algorithm = MOEAD(
        problem=problem,
        population_size=300,
        crossover=DifferentialEvolutionCrossover(CR=1.0, F=0.5, K=0.5),
        mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20),
        aggregative_function=Tschebycheff(dimension=problem.number_of_objectives),
        neighbor_size=20,
        neighbourhood_selection_probability=0.9,
        max_number_of_replaced_solutions=2,
        weight_files_path='../../resources/MOEAD_weights',
        termination_criterion=StoppingByEvaluations(max=max_evaluations)
    )

    algorithm.observable.register(observer=ProgressBarObserver(max=max_evaluations))

    algorithm.run()
    front = algorithm.get_result()
