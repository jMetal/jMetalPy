import logging
from concurrent.futures import ProcessPoolExecutor

jMetalPyLogger = logging.getLogger('jMetalPy')

"""
.. module:: laboratory
   :platform: Unix, Windows
   :synopsis: Run experiments. WIP!

.. moduleauthor:: Antonio Benítez-Hidalgo <antonio.b@uma.es>
"""


def experiment(algorithm_list: list, metric_list: list, problem_list: list, g_params: dict=None, m_workers: int=3):
    """ :param algorithm_list: List of algorithms as Tuple(Algorithm, dic() with parameters).
    :param metric_list: List of metrics.
    :param problem_list:  List of problems as Tuple(Problem, dic() with parameters).
    :param g_params: Global parameters (will override those from algorithm_list).
    :param m_workers: Maximum number of workers for ProcessPoolExecutor.
    :return: Stats.
    """

    with ProcessPoolExecutor(max_workers=m_workers) as pool:
        result = dict()

        for p_index, (problem, problem_params) in enumerate(problem_list):
            if isinstance(problem, type):
                jMetalPyLogger.debug('Problem is not instantiated by default')
                problem = problem(**problem_params)

            for a_index, (algorithm, algorithm_params) in enumerate(algorithm_list):
                if g_params:
                    algorithm_params.update(g_params)
                if isinstance(algorithm, type):
                    jMetalPyLogger.debug('Algorithm {} is not instantiated by default'.format(algorithm))
                    algorithm_list[a_index] = (algorithm(problem=problem, **algorithm_params), {})

                jMetalPyLogger.info('Running experiment: problem {0}, algorithm {1}'.format(problem, algorithm))

                pool.submit(algorithm_list[a_index][0].run())

            jMetalPyLogger.debug('Waiting')

        # Wait until all computation is done for this problem
        pool.shutdown(wait=True)

    for algorithm, _ in algorithm_list:
        front = algorithm.get_result()
        result[algorithm.get_name()] = {'front': front,
                                        'problem': algorithm.problem.get_name(),
                                        'time': algorithm.total_computing_time}

        for metric in metric_list:
            result[algorithm.get_name()].setdefault('metric', dict()).update({metric.get_name(): metric.compute(front)})

    return result


def display(table: dict):
    for k, v in table.items():
        print('{0}: {1}'.format(k, v['metric']))
