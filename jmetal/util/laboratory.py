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

    def __init__(self, algorithm_list: list, n_runs: int = 1, m_workers: int = 6):
        """ :param algorithm_list: List of algorithms as Tuple(Algorithm, dic() with parameters).
        :param m_workers: Maximum number of workers for ProcessPoolExecutor. """
        self.algorithm_list = algorithm_list

        self.n_runs = n_runs
        self.m_workers = m_workers
        self.experiment_list = list()

    def run(self) -> None:
        """ Run the experiment. """
        self.__configure_algorithm_list()

        with ProcessPoolExecutor(max_workers=self.m_workers) as pool:
            for name, algorithm, n_run in self.experiment_list:
                jMetalPyLogger.info('Running experiment {0}'.format(
                    name, algorithm.problem, n_run)
                )

                pool.submit(algorithm.run())

            # Wait until all computation is done for this problem
            jMetalPyLogger.debug('Waiting')
            pool.shutdown(wait=True)

    def compute_metrics(self, metric_list: list) -> dict:
        """ :param metric_list: List of metrics. Each metric should inherit from :py:class:`Metric` or, at least,
        contain a method `compute`. """
        results = dict()

        for name, algorithm, n_run in self.experiment_list:
            results[name] = {}

            for metric in metric_list:
                results[name].setdefault('metric', dict()).update(
                    {metric.get_name(): metric.compute(algorithm.get_result())}
                )

        return results

    def export_to_file(self, base_directory: str = 'experiment', function_values_filename: str = 'FUN',
                       variables_filename: str = 'VAR'):
        for tag, algorithm, run in self.experiment_list:
            # todo Save VAR and FUN to files
            pass

    def __configure_algorithm_list(self):
        """ Configure the algorithm list, by making a triple of (name, algorithm, run). """
        for n_run in range(self.n_runs):
            for tag, algorithm in self.algorithm_list:
                name = '{0}.{1}.{2}'.format(tag, algorithm.problem.get_name(), n_run)
                self.experiment_list.append((name, algorithm, n_run))
