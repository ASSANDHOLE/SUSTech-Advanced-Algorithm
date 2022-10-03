import random
from copy import deepcopy
from typing import List, Tuple, Any, Callable, Type

import numpy as np
import matplotlib.pyplot as plt

from algorithms import load_balancing_int, load_balancing_greedy_int
from algorithms import load_balancing_float, load_balancing_greedy_float
from algorithms import load_balancing_diff_exec_time_int, load_balancing_diff_exec_time_float

from algorithms import simple_pso

from hw03 import get_random_color, plot_load_balancing_bar_chat


def load_balancing_sorted(tasks: np.ndarray, worker_num: int) -> Tuple[List[List[int | float]], float | int]:
    tasks = -np.sort(-tasks)
    workers = np.zeros(worker_num, dtype=tasks.dtype)
    worker_load = [[] for _ in range(worker_num)]
    for job in tasks:
        least_worker = np.argmin(workers)
        workers[least_worker] += job
        worker_load[least_worker].append(job)
    return worker_load, np.max(workers)


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
    arr = [10.0, 5.156251049090973, 10.0, 0.8291241115149064, 2.064953172395254, 4.843750889022266, 0.01,
           5.485400293053465, 1.6104065463335664]
    return np.array(arr, dtype=np.float32), 4


def task2_1() -> Tuple[np.ndarray, int]:
    """
    Task 2.1

    Returns
    Tuple[np.ndarray, int]
        The jobs and the worker number.
    """
    arr = [1, 1, 2, 2, 3, 3]
    return np.array(arr, dtype=np.int32), 3


def task2_2() -> Tuple[np.ndarray, int]:
    """
    Task 2.2

    Returns
    Tuple[np.ndarray, int]
        The jobs and the worker number.
    """
    arr = [10.0, 5.15625576208762, 10.0, 0.8291843983168172, 2.0646359481368566, 4.843785836097282, 0.01,
           5.4853783834758065, 1.6104999077668771]
    return np.array(arr, dtype=np.float32), 4


def task3_1() -> List[list]:
    """
    Task 3.1

    Returns
    -------
    List[list]
        The jobs execution time for each worker.
    """
    jobs = [list(range(1, 16)),
            [i * 2 for i in range(1, 16)],
            [i * 3 > 30 and 15 or i * 3 for i in range(1, 16)]]
    return jobs


def task3_naive(jobs: List[list], dtype: Type) -> Tuple[List[list], int | float]:
    worker_num = len(jobs)
    func = dtype == np.int32 and load_balancing_diff_exec_time_int or load_balancing_diff_exec_time_float
    jobs_ndarray = np.array(jobs, dtype=dtype)
    load, val = func(jobs_ndarray, worker_num)
    return load, val


def task3_algo(jobs: List[list]) -> Tuple[List[list], int | float]:
    """
    Task 3's algorithm.

    Parameters
    ----------
    jobs : List[list]
        The jobs execution time for each worker.

    Returns
    -------
    Tuple[List[list], int | float]
        The worker loads and the max load.
    """
    worker_num = len(jobs)
    job_num = len(jobs[0])
    ratio = [[] for _ in range(worker_num)]
    for i in range(job_num):
        min_exec_time = min([jobs[j][i] for j in range(worker_num)])
        for j in range(worker_num):
            ratio[j].append((jobs[j][i] - min_exec_time, jobs[j][i] / min_exec_time))

    execd_job_id = []
    job_exec_priority = [list(range(job_num)) for _ in range(worker_num)]
    for i in range(worker_num):
        job_exec_priority[i].sort(key=lambda x: (*ratio[i][x], -jobs[i][x]))

    worker_load = [[] for _ in range(worker_num)]
    worker_exec_time = [0 for _ in range(worker_num)]
    while len(execd_job_id) < job_num:
        least_worker = np.argmin(worker_exec_time)
        for job_id in job_exec_priority[least_worker]:
            if job_id not in execd_job_id:
                worker_load[least_worker].append(jobs[least_worker][job_id])
                worker_exec_time[least_worker] += jobs[least_worker][job_id]
                execd_job_id.append(job_id)
                break

    return worker_load, np.max(worker_exec_time)


def random_job_int():
    arr = np.random.randint(1, 10, 7)
    return arr, 3


def random_job_float():
    arr = np.random.rand(7).astype(np.float32)
    return arr, 3


def pso_minimize_avg_cost(jobs):
    jobs = np.array(jobs, dtype=np.float32)
    worker_num = 4
    _, val = load_balancing_float(jobs, worker_num)
    _, _, avg_time = load_balancing_greedy_float(jobs, worker_num)
    return -(avg_time / val)


def pso_minimize_sorted_cost(jobs):
    jobs = np.array(jobs, dtype=np.float32)
    worker_num = 4
    _, _, avg_time = load_balancing_greedy_float(jobs, worker_num)
    sorted_load, sorted_val = load_balancing_sorted(jobs, worker_num)
    return -(avg_time / sorted_val)


def pso_over_pso(initial):
    bounds = [(0.01, 10) for _ in range(len(initial))]
    d, _ = simple_pso.minimize(pso_minimize_avg_cost, initial, bounds, num_particles=30, maxiter=80, verbose=False)
    return d


def main_pso(cost_func):
    initial = [[10.0, 5.15625576208762, 10.0, 0.8291843983168172, 2.0646359481368566, 4.843785836097282, 0.01,
                5.4853783834758065, 1.6104999077668771]]
    initial_all = [[random.random() * 9.99 + 0.01 for _ in range(9)] for _ in range(47)]
    initial_all += initial
    bounds = [(0.01, 10) for _ in range(len(initial[0]))]
    simple_pso.minimize_custom(cost_func, initial_all, bounds, num_particles=50, maxiter=100, verbose=True)


def function_getter(tasks) -> Tuple[Callable, Callable]:
    if tasks.dtype == np.int32:
        return load_balancing_int, load_balancing_greedy_int
    else:
        return load_balancing_float, load_balancing_greedy_float


def main_random_task2():
    while True:
        tasks, worker_num = random_job_int()
        lb_func, lb_greedy_func = function_getter(tasks)
        opt_load, opt_val = lb_func(tasks, worker_num)
        sorted_load, sorted_val = load_balancing_sorted(tasks, worker_num)
        (best_load, best_val), (worst_load, worst_val), avg_time \
            = lb_greedy_func(tasks, worker_num)
        if abs(sorted_val - opt_val) < 1e-6 and avg_time > sorted_val:
            break
    print(f'{opt_val=}')
    print(f'{best_val=}')
    print(f'{sorted_val=}')
    print(f'{worst_val=}')
    print(f'{avg_time=}')
    print(f'avg/sorted={avg_time / sorted_val}')


def main_task1():
    tasks, worker_num = task1_2()
    lb_func, lb_greedy_func = function_getter(tasks)
    opt_load, opt_val = lb_func(tasks, worker_num)
    (best_load, best_val), (worst_load, worst_val), avg_time \
        = lb_greedy_func(tasks, worker_num)
    print(f'{opt_val=}')
    print(f'{best_val=}')
    print(f'{worst_val=}')
    print(f'{avg_time=}')
    print(f'avg/t*={avg_time / opt_val}')


def main_task2():
    tasks, worker_num = task2_1()
    lb_func, lb_greedy_func = function_getter(tasks)
    opt_load, opt_val = lb_func(tasks, worker_num)
    sorted_load, sorted_val = load_balancing_sorted(tasks, worker_num)
    (best_load, best_val), (worst_load, worst_val), avg_time \
        = lb_greedy_func(tasks, worker_num)
    print(f'{opt_val=}')
    print(f'{best_val=}')
    print(f'{sorted_val=}')
    print(f'{worst_val=}')
    print(f'{avg_time=}')
    print(f'avg/sorted={avg_time / sorted_val}')
    plot_load_balancing_bar_chat(sorted_load)


def main_task3():
    jobs = task3_1()
    # load, val = task3_algo(jobs)
    load, val = task3_naive(jobs, np.int32)
    print(f'{val=}')
    plot_load_balancing_bar_chat(load)


if __name__ == '__main__':
    main_task3()
