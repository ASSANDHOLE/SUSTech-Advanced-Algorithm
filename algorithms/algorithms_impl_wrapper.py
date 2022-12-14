import os
import sys

from ctypes import *
from struct import pack, unpack
from typing import Tuple, List, Literal, Type

import numpy as np
from numpy.ctypeslib import ndpointer as numpy_nd_ptr
from numpy.ctypeslib import as_array


def _get_dll_name():
    base_name = 'libalgo_cpp_impl'
    if sys.platform == 'linux':
        dll_name = f'{base_name}.so'
    elif sys.platform == 'win32':
        dll_name = f'{base_name}.dll'
    else:
        # Your system's dll name
        # dll_name = None
        raise NotImplementedError(f'Your system is not registered: {sys.platform}')
    return dll_name


_dll_name = _get_dll_name()
_lib = cdll.LoadLibrary(os.path.join(os.path.dirname(__file__), _dll_name))

i32 = np.int32
f32 = np.float32


def nd_ptr(data_type: Type):
    return numpy_nd_ptr(dtype=data_type, flags='aligned, c_contiguous')


def _c_tsp_naive():
    tspn = _lib.TravellingSalesmanNaive
    tspn.restype = POINTER(c_int)
    tspn.argtypes = [nd_ptr(f32), c_int]
    return tspn


def _c_load_balancing_int():
    lb = _lib.LoadBalancingInt
    lb.restype = POINTER(c_int)
    lb.argtypes = [nd_ptr(i32), c_int, c_int]
    return lb


def _c_load_balancing_greedy_int():
    lb = _lib.LoadBalancingGreedyInt
    lb.restype = POINTER(c_int)
    lb.argtypes = [nd_ptr(i32), c_int, c_int]
    return lb


def _c_load_balancing_float():
    lb = _lib.LoadBalancingFloat
    lb.restype = POINTER(c_float)
    lb.argtypes = [nd_ptr(f32), c_int, c_int]
    return lb


def _c_load_balancing_greedy_float():
    lb = _lib.LoadBalancingGreedyFloat
    lb.restype = POINTER(c_float)
    lb.argtypes = [nd_ptr(f32), c_int, c_int]
    return lb


def _c_load_balancing_diff_exec_time_int():
    lb = _lib.LoadBalancingDifferentExecTimeInt
    lb.restype = POINTER(c_int)
    lb.argtypes = [nd_ptr(i32), c_int, c_int]
    return lb


def _c_load_balancing_diff_exec_time_float():
    lb = _lib.LoadBalancingDifferentExecTimeFloat
    lb.restype = POINTER(c_float)
    lb.argtypes = [nd_ptr(f32), c_int, c_int]
    return lb


def _c_center_selection_int():
    cs = _lib.CenterSelectionInt
    cs.restype = POINTER(c_int)
    cs.argtypes = [nd_ptr(i32), c_int, c_int]
    return cs


def _c_center_selection_float():
    cs = _lib.CenterSelectionFloat
    cs.restype = POINTER(c_float)
    cs.argtypes = [nd_ptr(f32), c_int, c_int]
    return cs


def _c_kmeans_2d_int():
    kmeans = _lib.KMeans2dInt
    kmeans.restype = POINTER(c_float)
    kmeans.argtypes = [nd_ptr(i32),
                       c_int, c_int,
                       c_int, c_float,
                       c_int]
    return kmeans


def _c_kmeans_2d_float():
    kmeans = _lib.KMeans2dFloat
    kmeans.restype = POINTER(c_float)
    kmeans.argtypes = [nd_ptr(f32),
                       c_int, c_int, c_int, c_float, c_int]
    return kmeans


def _c_kmedoids_2d_int():
    kmedoids = _lib.KMedoids2dInt
    kmedoids.restype = POINTER(c_int)
    kmedoids.argtypes = [nd_ptr(i32),
                         c_int, c_int,
                         c_int, c_int]
    return kmedoids


def _c_kmedoids_2d_float():
    kmedoids = _lib.KMedoids2dFloat
    kmedoids.restype = POINTER(c_int)
    kmedoids.argtypes = [nd_ptr(f32),
                         c_int, c_int,
                         c_int, c_int]
    return kmedoids


def _c_fuzzy_c_means_2d_int():
    fcm = _lib.FuzzyCMeans2dInt
    fcm.restype = POINTER(c_float)
    fcm.argtypes = [nd_ptr(i32),
                    c_int, c_int,
                    c_float, c_int,
                    c_float, c_int]
    return fcm


def _c_fuzzy_c_means_2d_float():
    fcm = _lib.FuzzyCMeans2dFloat
    fcm.restype = POINTER(c_float)
    fcm.argtypes = [nd_ptr(f32),
                    c_int, c_int,
                    c_float, c_int,
                    c_float, c_int]
    return fcm


def _c_set_cover_int():
    sc = _lib.SetCoverInt
    sc.restype = POINTER(c_int)
    sc.argtypes = [nd_ptr(i32),
                   nd_ptr(i32),
                   nd_ptr(i32),
                   nd_ptr(i32),
                   c_int]
    return sc


def _c_set_cover_float():
    sc = _lib.SetCoverFloat
    sc.restype = POINTER(c_float)
    sc.argtypes = [nd_ptr(i32),
                   nd_ptr(i32),
                   nd_ptr(f32),
                   nd_ptr(i32),
                   c_int]
    return sc


def _c_pricing_method_int():
    pmf = _lib.PricingMethodInt
    pmf.restype = POINTER(c_int)
    pmf.argtypes = [
        nd_ptr(i32),
        c_int,
        nd_ptr(i32),
        c_int,
        nd_ptr(i32),
        nd_ptr(i32),
    ]
    return pmf


def _c_pricing_method_float():
    pmf = _lib.PricingMethodFloat
    pmf.restype = POINTER(c_float)
    pmf.argtypes = [
        nd_ptr(i32),
        c_int,
        nd_ptr(f32),
        c_int,
        nd_ptr(i32),
        nd_ptr(i32),
    ]
    return pmf


def _c_disjoint_path_problem_c1():
    cdp = _lib.DisjointPathProblemC1
    cdp.restype = POINTER(c_int)
    cdp.argtypes = [
        c_int,
        nd_ptr(i32),
        c_int,
        nd_ptr(i32),
        nd_ptr(i32),
        c_int
    ]
    return cdp


def _c_disjoint_path_problem_cn():
    cdp = _lib.DisjointPathProblemCn
    cdp.restype = POINTER(c_int)
    cdp.argtypes = [
        c_int,
        nd_ptr(i32),
        c_int,
        nd_ptr(i32),
        nd_ptr(i32),
        c_int,
        c_int
    ]
    return cdp


_tsp_naive = _c_tsp_naive()
_load_balancing_int = _c_load_balancing_int()
_load_balancing_greedy_int = _c_load_balancing_greedy_int()
_load_balancing_float = _c_load_balancing_float()
_load_balancing_greedy_float = _c_load_balancing_greedy_float()
_load_balancing_diff_exec_time_int = _c_load_balancing_diff_exec_time_int()
_load_balancing_diff_exec_time_float = _c_load_balancing_diff_exec_time_float()
_center_selection_int = _c_center_selection_int()
_center_selection_float = _c_center_selection_float()
_kmeans_2d_int = _c_kmeans_2d_int()
_kmeans_2d_float = _c_kmeans_2d_float()
_kmedoids_2d_int = _c_kmedoids_2d_int()
_kmedoids_2d_float = _c_kmedoids_2d_float()
_fuzzy_c_means_2d_int = _c_fuzzy_c_means_2d_int()
_fuzzy_c_means_2d_float = _c_fuzzy_c_means_2d_float()
_set_cover_int = _c_set_cover_int()
_set_cover_float = _c_set_cover_float()
_pricing_method_int = _c_pricing_method_int()
_pricing_method_float = _c_pricing_method_float()
_disjoint_path_problem_c1 = _c_disjoint_path_problem_c1()
_disjoint_path_problem_cn = _c_disjoint_path_problem_cn()


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


def load_balancing_greedy_int(jobs: np.ndarray, worker_num: int) -> Tuple[
    Tuple[List[List[int]], int], Tuple[List[List[int]], int], float]:
    """
    Parameters
    ----------
    jobs : np.ndarray
        The jobs to distribute.
    worker_num : int
        The number of workers.

    Returns
    -------
    Tuple[Tuple[List[List[int]], int], Tuple[List[List[int]], int], float]
        The best load of each worker and the maximum load value,
        and the worst load of each worker and the maximum load value.
        The last float is the average load value.
    """
    jobs = jobs.astype(np.int32)
    res = _load_balancing_greedy_int(jobs, len(jobs), worker_num)
    res = as_array(res, shape=(2 * len(jobs) + 1,))
    best_res = _load_balancing_greedy_helper(res[:len(jobs)], worker_num)
    worst_res = _load_balancing_greedy_helper(res[len(jobs):-1], worker_num)
    avg_time = res[-1]
    avg_time = pack('i', avg_time)
    avg_time = unpack('f', avg_time)[0]
    return best_res, worst_res, avg_time


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


def load_balancing_greedy_float(jobs: np.ndarray, worker_num: int) -> Tuple[
    Tuple[List[List[float]], float], Tuple[List[List[float]], float], float]:
    """
    Parameters
    ----------
    jobs : np.ndarray
        The jobs to distribute.
    worker_num : int
        The number of workers.

    Returns
    -------
    Tuple[Tuple[List[List[float]], float], Tuple[List[List[float]], float], float]
        The best load of each worker and the maximum load value,
        and the worst load of each worker and the maximum load value.
        The last float is the average load value.
    """
    jobs = jobs.astype(np.float32)
    res = _load_balancing_greedy_float(jobs, len(jobs), worker_num)
    res = as_array(res, shape=(2 * len(jobs) + 1,))
    best_res = _load_balancing_greedy_helper(res[:len(jobs)], worker_num)
    worst_res = _load_balancing_greedy_helper(res[len(jobs):-1], worker_num)
    return best_res, worst_res, res[-1]


def load_balancing_diff_exec_time_int(jobs: np.ndarray, worker_num: int) -> Tuple[List[List[int]], int]:
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
    jobs = jobs.astype(np.int32).flatten('C')
    res = _load_balancing_diff_exec_time_int(jobs, len(jobs), worker_num)
    res = as_array(res, shape=(2 * worker_num + len(jobs),))
    res_arr = [[] for _ in range(worker_num)]
    start = worker_num
    max_time = 0
    for i in range(worker_num):
        end = start + res[i]
        res_arr[i].extend([*res[start:end]])
        max_time = max(max_time, sum(res_arr[i]))
        start = end
    worker_order = res[start:]
    # sort the res_arr according to the worker_order
    res_arr = [res_arr[i] for i in np.argsort(worker_order)]
    return res_arr, max_time


def load_balancing_diff_exec_time_float(jobs: np.ndarray, worker_num: int) -> Tuple[List[List[float]], float]:
    """
    Parameters
    ----------
    jobs : np.ndarray
        The jobs to distribute.
    worker_num : int
        The number of workers.

    Returns
    -------
    Tuple[List[List[float]], float]
        The load of each worker and the maximum load value.
    """
    jobs = jobs.astype(np.float32).flatten('C')
    res = _load_balancing_diff_exec_time_float(jobs, len(jobs), worker_num)
    res = as_array(res, shape=(2 * worker_num + len(jobs),))
    res_arr = [[] for _ in range(worker_num)]
    start = worker_num
    max_time = 0
    for i in range(worker_num):
        end = start + res[i]
        res_arr[i].extend([*res[start:end]])
        max_time = max(max_time, sum(res_arr[i]))
        start = end
    worker_order = res[start:]
    # sort the res_arr according to the worker_order
    pk = pack('f' * len(worker_order), *worker_order)
    worker_order = list(unpack('i' * len(worker_order), pk))
    res_arr = [res_arr[i] for i in np.argsort(worker_order)]
    return res_arr, max_time


def center_selection_dist(points: np.ndarray, centers: np.ndarray | List[np.ndarray]) -> float:
    dist = np.linalg.norm(points - centers[0], axis=1)
    for i in range(1, len(centers)):
        dist = np.minimum(dist, np.linalg.norm(points - centers[i], axis=1))
    return np.max(dist)


def center_selection_int(points: np.ndarray, center_num: int) -> Tuple[float, np.ndarray]:
    """
    Parameters
    ----------
    points : np.ndarray
        The points to select.
    center_num : int
        The number of centers.

    Returns
    -------
    Tuple[float, np.ndarray]
        The maximum distance to any center and the centers.
    """
    points = points.astype(np.int32)
    res = _center_selection_int(points, points.shape[0], center_num)
    res = as_array(res, shape=(center_num, 2))
    dist = center_selection_dist(points, res)
    return dist, res


def center_selection_float(points: np.ndarray, center_num: int) -> Tuple[float, np.ndarray]:
    """
    Parameters
    ----------
    points : np.ndarray
        The points to select.
    center_num : int
        The number of centers.

    Returns
    -------
    Tuple[float, np.ndarray]
        The maximum distance to any center and the centers.
    """
    points = points.astype(np.float32)
    res = _center_selection_float(points, points.shape[0], center_num)
    res = as_array(res, shape=(center_num, 2))
    dist = center_selection_dist(points, res)
    return dist, res


__KMEANS_INIT_FN_NAME_ID_MAP = {
    'random': 0,
    'kmeans++': 1,
}


def _kmeans_get_labels(points: np.ndarray, centers: np.ndarray) -> np.ndarray:
    if points.dtype != np.float32:
        points = points.astype(np.float32)
    dist = np.linalg.norm(points - centers[0], axis=1)
    labels = np.zeros(len(points), dtype=np.int32)
    for i in range(1, len(centers)):
        dist_new = np.linalg.norm(points - centers[i], axis=1)
        labels[dist_new < dist] = i
        dist = np.minimum(dist, dist_new)
    return labels


def kmeans_2d_int(points: np.ndarray, k: int,
                  init_fn: Literal['kmeans++', 'random'] = 'kmeans++',
                  eps: float = 1e-6, max_iter: int = 100
                  ) -> Tuple[np.ndarray, np.ndarray, int, bool]:
    """
    Parameters
    ----------
    points : np.ndarray
        The points to cluster.
    k : int
        The number of clusters.
    init_fn : Literal['kmeans++', 'random']
        The initialization function.
    eps : float
        The convergence threshold.
    max_iter : int
        The maximum number of iterations.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, int, bool]
        The cluster centers, the cluster labels,
         the number of iterations, and whether the algorithm converged.
    """

    points = points.astype(np.int32)
    init_fn_id = __KMEANS_INIT_FN_NAME_ID_MAP[init_fn]
    res = _kmeans_2d_int(points, points.shape[0], k, init_fn_id, eps, max_iter)
    res = as_array(res, shape=(k * 2 + 2,))
    centers = res[:k * 2].reshape((k, 2))
    labels = _kmeans_get_labels(points, centers)
    n_iter = res[-2]
    n_iter = pack('f', n_iter)
    n_iter = unpack('i', n_iter)[0]
    converged = not (abs(res[-1]) < 0.001)
    return centers, labels, n_iter, converged


def kmeans_2d_float(points: np.ndarray, k: int,
                    init_fn: Literal['kmeans++', 'random'] = 'kmeans++',
                    eps: float = 1e-6, max_iter: int = 100
                    ) -> Tuple[np.ndarray, np.ndarray, int, bool]:
    """
    Parameters
    ----------
    points : np.ndarray
        The points to cluster.
    k : int
        The number of clusters.
    init_fn : Literal['kmeans++', 'random']
        The initialization function.
    eps : float
        The convergence threshold.
    max_iter : int
        The maximum number of iterations.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, int, bool]
        The cluster centers, the cluster labels,
         the number of iterations, and whether the algorithm converged.
    """

    points = points.astype(np.float32)
    init_fn_id = __KMEANS_INIT_FN_NAME_ID_MAP[init_fn]
    res = _kmeans_2d_float(points, points.shape[0], k, init_fn_id, eps, max_iter)
    res = as_array(res, shape=(k * 2 + 2,))
    centers = res[:k * 2].reshape((k, 2))
    labels = _kmeans_get_labels(points, centers)
    n_iter = res[-2]
    n_iter = pack('f', n_iter)
    n_iter = unpack('i', n_iter)[0]
    converged = not (abs(res[-1]) < 0.001)
    return centers, labels, n_iter, converged


def square_euclidian_dist(points: np.ndarray, centers: np.ndarray, labels: np.ndarray | None = None) -> np.ndarray:
    """
    Parameters
    ----------
    points : np.ndarray
        The points to compute the distance.
    centers : np.ndarray
        The centers to compute the distance.
    labels : np.ndarray | None = None
        The labels of the points.

    Returns
    -------
    np.ndarray
        The square euclidian distance.
    """
    if points.dtype != np.float32:
        points = points.astype(np.float32)
    if centers.dtype != np.float32:
        centers = centers.astype(np.float32)
    if labels is None:
        labels = _kmeans_get_labels(points, centers)
    dist = np.linalg.norm(points - centers[labels], axis=1)
    return dist * dist


def _kmedoids_get_labels(points: np.ndarray, centers_idx: np.ndarray) -> np.ndarray:
    if points.dtype != np.float32:
        points = points.astype(np.float32)
    centers = points[centers_idx]
    return _kmeans_get_labels(points, centers)


def kmedoids_2d_int(points: np.ndarray, k: int,
                    init_fn: Literal['kmeans++', 'random'] = 'kmeans++',
                    max_iter: int = 100
                    ) -> Tuple[np.ndarray, np.ndarray, int, bool]:
    """
    Parameters
    ----------
    points : np.ndarray
        The points to cluster.
    k : int
        The number of clusters.
    init_fn : Literal['kmeans++', 'random']
        The initialization function.
    max_iter : int
        The maximum number of iterations.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, int, bool]
        The cluster center idx, the cluster labels,
         the number of iterations, and whether the algorithm converged.
    """

    points = points.astype(np.int32)
    init_fn_id = __KMEANS_INIT_FN_NAME_ID_MAP[init_fn]
    res = _kmedoids_2d_int(points, points.shape[0], k, init_fn_id, max_iter)
    res = as_array(res, shape=(k + 2,))
    centers_idx = res[:k]
    labels = _kmedoids_get_labels(points, centers_idx)
    n_iter = res[-2]
    converged = res[-1] != 0
    return centers_idx, labels, n_iter, converged


def kmedoids_2d_float(points: np.ndarray, k: int,
                      init_fn: Literal['kmeans++', 'random'] = 'kmeans++',
                      max_iter: int = 100
                      ) -> Tuple[np.ndarray, np.ndarray, int, bool]:
    """
        Parameters
        ----------
        points : np.ndarray
            The points to cluster.
        k : int
            The number of clusters.
        init_fn : Literal['kmeans++', 'random']
            The initialization function.
        max_iter : int
            The maximum number of iterations.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, int, bool]
            The cluster center idx, the cluster labels,
             the number of iterations, and whether the algorithm converged.
        """

    points = points.astype(np.float32)
    init_fn_id = __KMEANS_INIT_FN_NAME_ID_MAP[init_fn]
    res = _kmedoids_2d_float(points, points.shape[0], k, init_fn_id, max_iter)
    res = as_array(res, shape=(k + 2,))
    centers_idx = res[:k]
    labels = _kmedoids_get_labels(points, centers_idx)
    n_iter = res[-2]
    converged = res[-1] != 0
    return centers_idx, labels, n_iter, converged


def fuzzy_c_means_2d_int(points: np.ndarray, k: int,
                         m: float = 2.0,
                         init_fn: Literal['kmeans++', 'random'] = 'kmeans++',
                         eps: float = 1e-6, max_iter: int = 100
                         ) -> Tuple[np.ndarray, np.ndarray, int, bool]:
    """
    Parameters
    ----------
    points : np.ndarray
        The points to cluster.
    k : int
        The number of clusters.
    m : float
        The fuzziness.
    init_fn : Literal['kmeans++', 'random']
        The initialization function.
    eps : float
        The convergence threshold.
    max_iter : int
        The maximum number of iterations.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, int, bool]
        The cluster centers, the membership matrix,
         the number of iterations, and whether the algorithm converged.
    """

    points = points.astype(np.int32)
    init_fn_id = __KMEANS_INIT_FN_NAME_ID_MAP[init_fn]
    res = _fuzzy_c_means_2d_int(points, points.shape[0], k, m, init_fn_id, eps, max_iter)
    res = as_array(res, shape=(k * 2 + points.shape[0] * k + 2,))
    centers = res[:k * 2].reshape((k, 2))
    membership_matrix = res[k * 2:-2].reshape((points.shape[0], k))
    n_iter = res[-2]
    n_iter = pack('f', n_iter)
    n_iter = unpack('i', n_iter)[0]
    converged = not (abs(res[-1]) < 0.001)
    return centers, membership_matrix, n_iter, converged


def fuzzy_c_means_2d_float(points: np.ndarray, k: int,
                           m: float = 2.0,
                           init_fn: Literal['kmeans++', 'random'] = 'kmeans++',
                           eps: float = 1e-6, max_iter: int = 100
                           ) -> Tuple[np.ndarray, np.ndarray, int, bool]:
    """
    Parameters
    ----------
    points : np.ndarray
        The points to cluster.
    k : int
        The number of clusters.
    m : float
        The fuzziness.
    init_fn : Literal['kmeans++', 'random']
        The initialization function.
    eps : float
        The convergence threshold.
    max_iter : int
        The maximum number of iterations.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, int, bool]
        The cluster centers, the membership matrix,
         the number of iterations, and whether the algorithm converged.
    """

    points = points.astype(np.float32)
    init_fn_id = __KMEANS_INIT_FN_NAME_ID_MAP[init_fn]
    res = _fuzzy_c_means_2d_float(points, points.shape[0], k, m, init_fn_id, eps, max_iter)
    res = as_array(res, shape=(k * 2 + points.shape[0] * k + 2,))
    centers = res[:k * 2].reshape((k, 2))
    membership_matrix = res[k * 2:-2].reshape((points.shape[0], k))
    n_iter = res[-2]
    n_iter = pack('f', n_iter)
    n_iter = unpack('i', n_iter)[0]
    converged = not (abs(res[-1]) < 0.001)
    return centers, membership_matrix, n_iter, converged


def set_cover_int(subsets: List[List[int]], weights: List[int] | np.ndarray) -> Tuple[List[int], int]:
    """
    Parameters
    ----------
    subsets : List[List[int]]
        The subsets.
    weights : List[int] | np.ndarray
        The weights.

    Returns
    -------
    Tuple[List[int], int]
        The used subset indices, the total weight
    """

    flat_subsets = []
    for subset in subsets:
        flat_subsets.extend(subset)
    flat_subsets = np.array(flat_subsets, dtype=np.int32)
    subset_lens = np.array([len(subset) for subset in subsets], dtype=np.int32)
    order = np.zeros_like(subset_lens).astype(np.int32)
    weights = np.array(weights, dtype=np.int32)
    res = _set_cover_int(flat_subsets, subset_lens, weights, order, len(subsets))
    res = as_array(res, shape=(2,))
    total_weight = res[0]
    n_subsets_used = res[1]
    return order[:n_subsets_used].tolist(), total_weight


def set_cover_float(subsets: List[List[int]], weights: List[float] | np.ndarray) -> Tuple[List[int], float]:
    """
    Parameters
    ----------
    subsets : List[List[int]]
        The subsets.
    weights : List[float] | np.ndarray
        The weights.

    Returns
    -------
    Tuple[List[int], float]
        The used subset indices, the total weight
    """
    flat_subsets = []
    for subset in subsets:
        flat_subsets.extend(subset)
    flat_subsets = np.array(flat_subsets, dtype=np.int32)
    subset_lens = np.array([len(subset) for subset in subsets], dtype=np.int32)
    order = np.zeros_like(subset_lens).astype(np.int32)
    weights = np.array(weights, dtype=np.float32)
    res = _set_cover_float(flat_subsets, subset_lens, weights, order, len(subsets))
    res = as_array(res, shape=(2,))
    total_weight = res[0]
    n_subsets_used = res[1]
    n_subsets_used = pack('f', n_subsets_used)
    n_subsets_used = unpack('i', n_subsets_used)[0]
    return order[:n_subsets_used].tolist(), total_weight


def pricing_method_int(edges: List[List[int]] | np.ndarray,
                       weights: List[int] | np.ndarray) -> \
        Tuple[Tuple[np.ndarray, int], Tuple[np.ndarray, int]]:
    """
    Pricing Method for Weight in int, i.e., compare will use ==.

    Parameters
    ----------
    edges : List[List[int]] | np.ndarray
       The graph data. [[node1, node2], ...]
    weights : List[int] | np.ndarray
        The weights of the nodes. [weight1, weight2, ...]

    Returns
    -------
    Tuple[Tuple[np.ndarray, int], Tuple[np.ndarray, int]]
        The best and the worst:
            The index of the graph cover and the weight.
    """
    edge_in = np.array(edges).flatten().astype(i32)
    weights = np.array(weights, dtype=i32)
    best = np.zeros_like(weights).astype(i32)
    worst = np.zeros_like(weights).astype(i32)
    res = _pricing_method_int(edge_in, len(edges), weights, len(weights), best, worst)
    res = as_array(res, shape=(2,))
    best_weight = res[0]
    worst_weight = res[1]
    best = np.where(best == 1)[0]
    worst = np.where(worst == 1)[0]
    return (best, best_weight), (worst, worst_weight)


def pricing_method_float(edges: List[List[int]] | np.ndarray,
                         weights: List[float] | np.ndarray) -> \
            Tuple[Tuple[np.ndarray, float], Tuple[np.ndarray, float]]:
        """
        Pricing Method for Weight in float, i.e., compare will use abs(-) < 1e-6.

        Parameters
        ----------
        edges : List[List[int]] | np.ndarray
        The graph data. [[node1, node2], ...]
        weights : List[float] | np.ndarray
            The weights of the nodes. [weight1, weight2, ...]

        Returns
        -------
        Tuple[Tuple[np.ndarray, float], Tuple[np.ndarray, float]]
            The best and the worst:
                The index of the graph cover and the weight.
        """
        edge_in = np.array(edges).flatten().astype(i32)
        weights = np.array(weights, dtype=f32)
        best = np.zeros_like(weights).astype(i32)
        worst = np.zeros_like(weights).astype(i32)
        res = _pricing_method_float(edge_in, len(edges), weights, len(weights), best, worst)
        res = as_array(res, shape=(2,))
        best_weight = res[0]
        worst_weight = res[1]
        best = np.where(best == 1)[0]
        worst = np.where(worst == 1)[0]
        return (best, best_weight), (worst, worst_weight)


def disjoint_path_problem(edges: List[List[int]] | np.ndarray,
                          route_pairs: List[Tuple[int, int]] | np.ndarray,
                          edge_capacity: int) -> Tuple[
    Tuple[List[int], List[List[Tuple[int, int]]]],
    Tuple[List[int], List[List[Tuple[int, int]]]]
]:
    """
    Disjoint Path Problem.

    Parameters
    ----------
    edges : List[List[int]] | np.ndarray
        The graph data. [[node1, node2], ...]
    route_pairs : List[Tuple[int, int] | List[int, int]] | np.ndarray
        The route pairs. [(node1, node2), ...]
    edge_capacity : int
        The capacity of the edges.

    Returns
    -------
    Tuple[Tuple[List[int], List[List[Tuple[int, int]]]],
          Tuple[List[int], List[List[Tuple[int, int]]]]]
        The best and the worst:
            The order and the list of routes.
    """
    edges = np.array(edges)
    edge_len = len(edges)
    edges = edges.flatten().astype(i32)
    route_pairs = np.array(route_pairs).astype(i32)
    start_nodes = route_pairs[:, 0]
    end_nodes = route_pairs[:, 1]
    start_nodes = np.ascontiguousarray(start_nodes)
    end_nodes = np.ascontiguousarray(end_nodes)
    source_target_len = len(start_nodes)
    vertex_num = np.max(edges) + 1
    if edge_capacity == 1:
        ret = _disjoint_path_problem_c1(vertex_num,
                                        edges, edge_len,
                                        start_nodes,
                                        end_nodes,
                                        source_target_len)
    else:
        ret = _disjoint_path_problem_cn(vertex_num,
                                        edges, edge_len,
                                        start_nodes,
                                        end_nodes,
                                        source_target_len,
                                        edge_capacity)
    tmp_ret = as_array(ret, shape=(2,))
    best_path_len = tmp_ret[0] * 2
    worst_path_len = tmp_ret[1] * 2
    ret = as_array(ret, shape=(2 + 2 * (best_path_len + worst_path_len + source_target_len),))
    cur_size = 2
    best_order = ret[cur_size:cur_size + source_target_len]
    cur_size += source_target_len
    worst_order = ret[cur_size:cur_size + source_target_len]
    cur_size += source_target_len
    best_path = ret[cur_size:cur_size + best_path_len]
    cur_size += best_path_len
    worst_path = ret[cur_size:cur_size + worst_path_len]
    best_paths = [[] for _ in range(source_target_len)]
    worst_paths = [[] for _ in range(source_target_len)]
    head = 0
    for i in range(source_target_len - 1):
        if head < len(best_paths) and best_paths[head] != route_pairs[best_order[i]][0]:
            continue
        cur = np.where(best_path[head:] == route_pairs[best_order[i]][1])[0]
        if len(cur) == 0:
            continue
        cur = cur[0]
        best_paths[i] = best_path[head:head + cur + 1].reshape(-1, 2).tolist()
        head += cur + 1
    best_paths[-1] = best_path[head:].reshape(-1, 2).tolist()
    head = 0
    for i in range(source_target_len - 1):
        if head < len(worst_paths) and worst_paths[head] != route_pairs[worst_order[i]][0]:
            continue
        cur = np.where(worst_path[head:] == route_pairs[worst_order[i]][1])[0]
        if len(cur) == 0:
            continue
        cur = cur[0]
        worst_paths[i] = worst_path[head:head + cur + 1].reshape(-1, 2).tolist()
        head += cur + 1
    worst_paths[-1] = worst_path[head:].reshape(-1, 2).tolist()
    return (best_order.tolist(), best_paths), (worst_order.tolist(), worst_paths)


