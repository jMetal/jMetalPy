import logging
from concurrent.futures import ProcessPoolExecutor

jMetalPyLogger = logging.getLogger('jMetalPy')

"""
.. module:: laboratory
   :platform: Unix, Windows
   :synopsis: Run experiments. WIP!

.. moduleauthor:: Antonio Benítez-Hidalgo <antonio.b@uma.es>
"""


class Experiment:

    def __init__(self, algorithm_list: list, problem_list: list, n_runs: int = 1, m_workers: int = 1):
        """ :param algorithm_list: List of algorithms as Tuple(Algorithm, dic() with parameters).
        :param problem_list:  List of problems as Tuple(Problem, dic() with parameters).
        :param m_workers: Maximum number of workers for ProcessPoolExecutor. """
        self.algorithm_list = algorithm_list
        self.problem_list = problem_list

        self.n_runs = n_runs
        self.m_workers = m_workers
        self.experiments_list = list()

    def run(self) -> None:
        """ Run the experiment. """
        self.__configure_algorithm_list()

        with ProcessPoolExecutor(max_workers=self.m_workers) as pool:
            for algorithm, problem, run in self.experiments_list:
                jMetalPyLogger.info('Running experiment: algorithm {0}, problem {1} (run {2})'.format(
                    algorithm.get_name(), problem.get_name(), run)
                )

                pool.submit(algorithm.run())

            # Wait until all computation is done for this problem
            jMetalPyLogger.debug('Waiting')
            pool.shutdown(wait=True)

    def compute_metrics(self, metric_list: list) -> None:
        """ :param metric_list: List of metrics. Each metric should inherit from :py:class:`Metric` or, at least,
        contain a method `compute`. """
        results = dict()

        for algorithm, problem, run in self.experiments_list:
            name = '{0}.{1}.{2}'.format(algorithm.__class__.__name__, problem.__class__.__name__, run)

            counter = 0
            while name in results:
                counter += 1
                name = '{0}.{1}.{2}({3})'.format(algorithm.__class__.__name__, problem.__class__.__name__, run, counter)

            results[name] = {}

            for metric in metric_list:
                results[name].setdefault('metric', dict()).update(
                    {metric.get_name(): metric.compute(algorithm.get_result())}
                )

        print(results)

    def export_to_file(self, base_directory: str = 'experiment', function_values_filename: str = 'FUN',
                       variables_filename: str = 'VAR'):
        for algorithm, problem, run in self.experiments_list:
            # todo Save VAR and FUN to files
            pass

    def __configure_algorithm_list(self):
        """ Configure the algorithm list, by making a triple of (algorithm, problem, run). """
        for n_run in range(self.n_runs):

            for p_index, (problem, problem_params) in enumerate(self.problem_list):
                if isinstance(problem, type):
                    jMetalPyLogger.debug('Problem {} is not instantiated by default'.format(problem))
                    problem = problem(**problem_params)

                for a_index, (algorithm, algorithm_params) in enumerate(self.algorithm_list):
                    if isinstance(algorithm, type):
                        jMetalPyLogger.debug('Algorithm {} is not instantiated by default'.format(algorithm))
                        algorithm = algorithm(problem=problem, **algorithm_params)

                    self.experiments_list.append((algorithm, problem, n_run))
