import os
import sys

from ctypes import *
from typing import Tuple, List

import numpy as np
from numpy.ctypeslib import ndpointer, as_array


def _get_dll_name():
    if sys.platform == 'linux':
        dll_name = 'libalgo_cpp_impl.so'
    elif sys.platform == 'win32':
        dll_name = 'libalgo_cpp_impl.dll'
    else:
        # Your system's dll name
        # dll_name = None
        raise NotImplementedError(f'Your system is not supported: {sys.platform}')
    return dll_name


_dll_name = _get_dll_name()
_lib = cdll.LoadLibrary(os.path.join(os.path.dirname(__file__), _dll_name))


def _c_tsp_naive():
    tspn = _lib.TravellingSalesmanNaive
    tspn.restype = POINTER(c_int)
    tspn.argtypes = [ndpointer(dtype=np.float32, flags='aligned, c_contiguous'), c_int]
    return tspn


def _c_load_balancing_int():
    lb = _lib.LoadBalancingInt
    lb.restype = POINTER(c_int)
    lb.argtypes = [ndpointer(dtype=np.int32, flags='aligned, c_contiguous'), c_int, c_int]
    return lb


def _c_load_balancing_greedy_int():
    lb = _lib.LoadBalancingGreedyInt
    lb.restype = POINTER(c_int)
    lb.argtypes = [ndpointer(dtype=np.int32, flags='aligned, c_contiguous'), c_int, c_int]
    return lb


def _c_load_balancing_float():
    lb = _lib.LoadBalancingFloat
    lb.restype = POINTER(c_float)
    lb.argtypes = [ndpointer(dtype=np.float32, flags='aligned, c_contiguous'), c_int, c_int]
    return lb


def _c_load_balancing_greedy_float():
    lb = _lib.LoadBalancingGreedyFloat
    lb.restype = POINTER(c_float)
    lb.argtypes = [ndpointer(dtype=np.float32, flags='aligned, c_contiguous'), c_int, c_int]
    return lb


_tsp_naive = _c_tsp_naive()
_load_balancing_int = _c_load_balancing_int()
_load_balancing_greedy_int = _c_load_balancing_greedy_int()
_load_balancing_float = _c_load_balancing_float()
_load_balancing_greedy_float = _c_load_balancing_greedy_float()


def tsp_naive(pts: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Parameters
    ----------
    pts : np.ndarray
        Points to visit.

    Returns
    -------
    np.ndarray
        The path.
    """
    dist = np.zeros((len(pts), len(pts)), dtype=np.float32, order='C')
    for i in range(len(pts)):
        for j in range(len(pts)):
            dist[i, j] = np.linalg.norm(pts[i] - pts[j])
    res = _tsp_naive(dist, len(pts))
    res = as_array(res, shape=(len(pts),))
    dist = 0
    old = 0
    for pt in res:
        dist += np.linalg.norm(pts[old] - pts[pt])
        old = pt
    dist += np.linalg.norm(pts[old] - pts[res[0]])
    return res, dist


def load_balancing_int(jobs: np.ndarray, worker_num: int) -> Tuple[List[List[int]], int]:
    """
    Parameters
    ----------
    jobs : np.ndarray
        The jobs to distribute.
    worker_num : int
        The number of workers.

    Returns
    -------
    Tuple[List[List[int]], int]
        The load of each worker and the maximum load value.
    """
    jobs = jobs.astype(np.int32)
    res = _load_balancing_int(jobs, len(jobs), worker_num)
    res = as_array(res, shape=(worker_num + len(jobs),))
    res_arr = [[] for _ in range(worker_num)]
    start = worker_num
    max_time = 0
    for i in range(worker_num):
        end = start + res[i]
        res_arr[i].extend([*res[start:end]])
        max_time = max(max_time, sum(res_arr[i]))
        start = end
    return res_arr, max_time


def _load_balancing_greedy_helper(jobs: np.ndarray, worker_num: int) -> Tuple[List[List[int]], int]:
    """
    Parameters
    ----------
    jobs : np.ndarray
        The jobs to distribute in certain order.
    worker_num : int
        The number of workers.

    Returns
    -------
    Tuple[List[List[int]], int]
        The load of each worker and the maximum load value in the order of jobs.
    """
    job_arr = [[] for _ in range(worker_num)]
    value_arr = np.array([0 for _ in range(worker_num)], dtype=jobs.dtype)
    for i in range(len(jobs)):
        min_idx = np.argmin(value_arr)
        job_arr[min_idx].append(jobs[i])
        value_arr[min_idx] += jobs[i]
    return job_arr, np.max(value_arr)


def load_balancing_greedy_int(jobs: np.ndarray, worker_num: int) -> Tuple[Tuple[List[List[int]], int], Tuple[List[List[int]], int]]:
    """
    Parameters
    ----------
    jobs : np.ndarray
        The jobs to distribute.
    worker_num : int
        The number of workers.

    Returns
    -------
    Tuple[Tuple[List[List[int]], int], Tuple[List[List[int]], int]]
        The best load of each worker and the maximum load value,
        and the worst load of each worker and the maximum load value.
    """
    jobs = jobs.astype(np.int32)
    res = _load_balancing_greedy_int(jobs, len(jobs), worker_num)
    res = as_array(res, shape=(2 * len(jobs),))
    best_res = _load_balancing_greedy_helper(res[:len(jobs)], worker_num)
    worst_res = _load_balancing_greedy_helper(res[len(jobs):], worker_num)
    return best_res, worst_res


def load_balancing_float(jobs: np.ndarray, worker_num: int) -> Tuple[List[List[float]], float]:
    """
    Parameters
    ----------
    jobs : np.ndarray
        The jobs to distribute in certain order.
    worker_num : int
        The number of workers.

    Returns
    -------
    Tuple[List[List[float]], float]
        The load of each worker and the maximum load value in the order of jobs.
    """
    jobs = jobs.astype(np.float32)
    res = _load_balancing_float(jobs, len(jobs), worker_num)
    res = as_array(res, shape=(worker_num + len(jobs),))
    res_arr = [[] for _ in range(worker_num)]
    start = worker_num
    max_time = 0
    for i in range(worker_num):
        end = start + int(res[i])
        res_arr[i].extend([*res[start:end]])
        max_time = max(max_time, sum(res_arr[i]))
        start = end
    return res_arr, max_time


def load_balancing_greedy_float(jobs: np.ndarray, worker_num: int) -> Tuple[Tuple[List[List[float]], float], Tuple[List[List[float]], float]]:
    """
    Parameters
    ----------
    jobs : np.ndarray
        The jobs to distribute.
    worker_num : int
        The number of workers.

    Returns
    -------
    Tuple[Tuple[List[List[float]], float], Tuple[List[List[float]], float]]
        The best load of each worker and the maximum load value,
        and the worst load of each worker and the maximum load value.
    """
    jobs = jobs.astype(np.float32)
    res = _load_balancing_greedy_float(jobs, len(jobs), worker_num)
    res = as_array(res, shape=(2 * len(jobs),))
    best_res = _load_balancing_greedy_helper(res[:len(jobs)], worker_num)
    worst_res = _load_balancing_greedy_helper(res[len(jobs):], worker_num)
    return best_res, worst_res
