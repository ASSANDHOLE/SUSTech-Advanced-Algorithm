from copy import deepcopy
from typing import List, Tuple, Any

import numpy as np
import matplotlib.pyplot as plt

from algorithms import load_balancing_int, load_balancing_greedy_int
from algorithms import load_balancing_float, load_balancing_greedy_float

from pso import pso_simple


def get_random_color() -> str:
    """
    Returns
    -------
    str
        A random color.
    """
    return f'#{np.random.randint(0, 0xFFFFFF):06X}'


def plot_load_balancing_bar_chat(val: List[List[float | int]], worker_label: List[str | Any] | None = None) -> None:
    """
    Parameters
    ----------
    val : List[List[float | int]]
        The values to plot.
    worker_label : List[str | Any] | None
        The labels of the workers. If None, the labels are the indices of the workers.
    """
    val = deepcopy(val)
    max_len = max([len(v) for v in val])
    for v in val:
        v.extend([0] * (max_len - len(v)))
    fig, ax = plt.subplots()
    starter = np.zeros(len(val))
    for i in range(max_len):
        if i == 0:
            b = ax.barh(range(len(val)), [v[0] for v in val])
            starter += [v[0] for v in val]
            for j in range(len(val)):
                b[j].set_color(get_random_color())
        else:
            b = ax.barh(range(len(val)), [v[i] for v in val], left=starter)
            starter += [v[i] for v in val]
            for j in range(len(val)):
                b[j].set_color(get_random_color())
    ax.set_yticks(range(len(val)))
    if worker_label is None:
        ax.set_yticklabels([f'Worker {i}' for i in range(len(val))])
    else:
        ax.set_yticklabels([str(x) for x in worker_label])
    ax.invert_yaxis()
    ax.set_xlabel('Load')
    ax.set_ylabel('Worker')
    ax.set_title('Load balancing')
    plt.show()


def task1_1() -> Tuple[np.ndarray, int]:
    """
    Task 1.1

    Returns
    Tuple[np.ndarray, int]
        The jobs and the worker number.
    """
    arr = [1, 1, 1, 1]
    return np.array(arr, dtype=np.int32), 2


def task1_2() -> Tuple[np.ndarray, int]:
    """
    Task 1.2

    Returns
    Tuple[np.ndarray, int]
        The jobs and the worker number.
    """
    arr = [4, 4, 4, 4, 1, 2]
    return np.array(arr, dtype=np.int32), 3


def task2_1() -> Tuple[np.ndarray, int]:
    """
    Task 1.2

    Returns
    Tuple[np.ndarray, int]
        The jobs and the worker number.
    """
    arr = [1, 1, 2, 2, 3, 3]
    return np.array(arr, dtype=np.int32), 3


def task2_2() -> Tuple[np.ndarray, int]:
    """
    Task 1.1

    Returns
    Tuple[np.ndarray, int]
        The jobs and the worker number.
    """
    arr = [3, 3, 5, 6, 7, 3, 1, 3, 4, 4]
    return np.array(arr, dtype=np.int32), 4


def task_3_1() -> Tuple[np.ndarray, int]:
    """
    Task 3.1

    Returns
    Tuple[np.ndarray, int]
        The jobs and the worker number.
    """
    arr = [0.589172449154356, 0.01, 0.1963623487745212, 0.38273428096229534, 0.3928805330488654, 0.7856225891891954, 0.4733953544903529, 0.11580742713733173, 0.196442091574372]
    arr = np.array(arr, dtype=np.float32)
    arr *= 10 / np.max(arr)
    return arr, 4


def main():
    jobs, worker_num = task_3_1()

    load, val = load_balancing_float(jobs, worker_num)
    (best_load, best_val), (worst_load, worst_val), *_ \
        = load_balancing_greedy_float(jobs, worker_num)

    print(f'optimal_val={val}')
    print(f'{best_val=}')
    print(f'{worst_val=}')
    print(f'ratio={worst_val / val}')
    print(f'{jobs=}')
    plot_load_balancing_bar_chat(load)
    plot_load_balancing_bar_chat(best_load)
    plot_load_balancing_bar_chat(worst_load)


def infinite_job_chart():
    n = 100
    jobs = [[1 / n for _ in range(n)] for _ in range(4)]
    jobs.insert(3, [0] * n)
    jobs.append([1])
    jobs[-1].extend([0] * (n - 1))
    label = ['Worker 1', 'Worker 2', 'Worker 3', '...', 'Worker n-1', 'Worker n']
    plot_load_balancing_bar_chat(jobs, worker_label=label)
    jobs = [[1 / n for _ in range(n - 1)] for _ in range(5)]
    jobs.insert(3, [0] * n)
    jobs[-1].append(1)
    plot_load_balancing_bar_chat(jobs, worker_label=label)


def random_job():
    arr = np.random.randint(1, 10, 6, np.int32)
    return arr, 3


def random_job_float():
    arr = np.random.rand(9).astype(np.float32)
    return arr, 4


def opt_out(rand_func, load_balancing_func, load_balancing_greedy_func, limit):
    while True:
        jobs, worker_num = rand_func()
        load, val = load_balancing_func(jobs, worker_num)
        (best_load, best_val), (worst_load, worst_val) \
            = load_balancing_greedy_func(jobs, worker_num)
        if worst_val > limit * val:
            # if jobs not the same, break
            print(f'optimal_val={val}')
            print(f'{best_val=}')
            print(f'{worst_val=:.4f}')
            print(f'{jobs=}')
            print(f'ratio={worst_val / val}')
            plot_load_balancing_bar_chat(load)
            plot_load_balancing_bar_chat(best_load)
            plot_load_balancing_bar_chat(worst_load)
            return jobs, worker_num, worst_val / val


def main_opt_int():
    opt_out(random_job, load_balancing_int, load_balancing_greedy_int, 1.75)


def main_opt_float():
    ratio = 1.2
    while True:
        _, _, ratio = opt_out(random_job_float, load_balancing_float, load_balancing_greedy_float, ratio)


def pso_minimize_cost(jobs):
    jobs = np.array(jobs, dtype=np.float32)
    worker_num = 4
    _, val = load_balancing_float(jobs, worker_num)
    _, (_, worst_val), *_ = load_balancing_greedy_float(jobs, worker_num)
    return -(worst_val / val)


def pso_opt_main():
    initial = [0.56425625, 0.06226046, 0.1495772, 0.0094707, 0.4112313,
               0.83713865, 0.5583354, 0.06189023, 0.27423367]
    bounds = [(0.01, 10) for _ in range(len(initial))]
    pso_simple.minimize(pso_minimize_cost, initial, bounds, num_particles=100, maxiter=100, verbose=True)


if __name__ == '__main__':
    main()
