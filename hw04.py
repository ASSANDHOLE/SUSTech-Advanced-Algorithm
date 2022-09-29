from copy import deepcopy
from typing import List, Tuple, Any

import numpy as np
import matplotlib.pyplot as plt

from algorithms import load_balancing_int, load_balancing_greedy_int
from algorithms import load_balancing_float, load_balancing_greedy_float

from pso import pso_simple

from hw03 import get_random_color, plot_load_balancing_bar_chat


def task1_1() -> Tuple[np.ndarray, int]:
    """
    Task 1.1

    Returns
    Tuple[np.ndarray, int]
        The jobs and the worker number.
    """
    arr = [1, 1, 2, 2, 3, 3]
    return np.array(arr, dtype=np.int32), 3


def task1_2() -> Tuple[np.ndarray, int]:
    """
    Task 1.2

    Returns
    Tuple[np.ndarray, int]
        The jobs and the worker number.
    """
    arr = [0.1630949095958792, 0.01422042186320435, 0.172311262975654, 0.673793348778232, 0.9974112035158145, 1.3470377916106362, 0.6732444050797997]
    return np.array(arr, dtype=np.float32), 3


def random_job_float():
    arr = np.random.rand(7).astype(np.float32)
    return arr, 3


def pso_minimize_avg_cost(jobs):
    jobs = np.array(jobs, dtype=np.float32)
    worker_num = 3
    _, val = load_balancing_float(jobs, worker_num)
    _, _, avg_time = load_balancing_greedy_float(jobs, worker_num)
    return -(avg_time / val)


def main_pso():
    initial = [1, 1, 2, 2, 3, 3]
    bounds = [(0.01, 10) for _ in range(len(initial))]
    pso_simple.minimize(pso_minimize_avg_cost, initial, bounds, num_particles=30, maxiter=80, verbose=True)


def main():
    tasks, worker_num = task1_2()
    if tasks.dtype == np.int32:
        lb_func = load_balancing_int
        lb_greedy_func = load_balancing_greedy_int
    else:
        lb_func = load_balancing_float
        lb_greedy_func = load_balancing_greedy_float
    opt_load, opt_val = lb_func(tasks, worker_num)
    (best_load, best_val), (worst_load, worst_val), avg_time \
        = lb_greedy_func(tasks, worker_num)
    print(f'{opt_val=}')
    print(f'{best_val=}')
    print(f'{worst_val=}')
    print(f'{avg_time=}')
    print(f'avg/t*={avg_time/opt_val}')


if __name__ == '__main__':
    main_pso()
