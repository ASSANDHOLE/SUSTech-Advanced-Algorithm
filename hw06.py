from copy import copy, deepcopy
from math import comb
from typing import Tuple, List, Literal
from functools import partial

import numpy as np
from matplotlib import pyplot as plt

from utils import create_animated_gif
from algorithms import center_selection_dist, center_selection_int, center_selection_float
from algorithms import simple_pso

from hw01 import get_circle_pts
from hw05 import draw_points
from hw05 import greedy_center_selection as dist_based_greedy_inclusion


def in_ndarray(arr: np.ndarray | List[np.ndarray], val: np.ndarray) -> bool:
    """
    Check if a value is in a numpy array.
    Parameters
    ----------
    arr : np.ndarray | List[np.ndarray]
        The numpy array.
    val : np.ndarray
        The value.

    Returns
    -------
    bool
        True if the value is in the array, False otherwise.
    """
    return any((val == a).all() for a in arr)


def dist_based_greedy_exclusion(pts: np.ndarray, center_num: int) -> Tuple[float, np.ndarray]:
    """
    Distance-based greedy removal algorithm.
    Parameters
    ----------
    pts : np.ndarray
        Points to visit.
    center_num : int
        Number of centers.

    Returns
    -------
    Tuple[float, np.ndarray]
        The objective value and the centers.
    """
    pts_old, pts = pts, deepcopy(pts)
    pairwise_dist = np.linalg.norm(pts[:, None] - pts, axis=2)
    while len(pts) > center_num:
        # select a pair of points with the minimum distance
        pairwise_dist[pairwise_dist == 0] = float('inf')
        min_dist = np.min(pairwise_dist)
        min_dist_idx = np.argwhere(pairwise_dist == min_dist)
        min_dist_idx = min_dist_idx[np.random.randint(len(min_dist_idx))]
        # select the point with the minimum distance to the pair
        closest_pt_val1 = np.sort(pairwise_dist[min_dist_idx[0]])[1]
        closest_pt_val2 = np.sort(pairwise_dist[min_dist_idx[1]])[1]
        which_pt = min_dist_idx[0] if closest_pt_val1 < closest_pt_val2 else min_dist_idx[1]
        # remove the point from the set of points
        pts = np.delete(pts, which_pt, axis=0)
        # update the pairwise distance matrix
        pairwise_dist = np.linalg.norm(pts[:, None] - pts, axis=2)
    return center_selection_dist(pts_old, pts), pts


def obj_based_greedy_inclusion(pts: np.ndarray, center_num: int) -> Tuple[float, np.ndarray]:
    """
    Greedy inclusion algorithm based on the original objective function.
    Parameters
    ----------
    pts : np.ndarray
        Points to visit.
    center_num : int
        Number of centers.

    Returns
    -------
    Tuple[float, np.ndarray]
        The objective value and the centers.
    """
    centers = []
    while len(centers) < center_num:
        best_obj = float('inf')
        best_pt = 0
        for i, pt in enumerate(pts):
            if in_ndarray(centers, pt):
                continue
            centers.append(pt)
            obj = center_selection_dist(pts, centers)
            if obj < best_obj:
                best_obj = obj
                best_pt = i
            centers.pop()
        centers.append(pts[best_pt])
    centers = np.array(centers)
    return center_selection_dist(pts, centers), centers


def obj_based_greedy_exclusion(pts: np.ndarray, center_num: int) -> Tuple[float, np.ndarray]:
    """
    Greedy removal algorithm based on the original objective function.
    Parameters
    ----------
    pts : np.ndarray
        Points to visit.
    center_num : int
        Number of centers.

    Returns
    -------
    Tuple[float, np.ndarray]
        The objective value and the centers.
    """
    pts_old, pts = pts, deepcopy(pts)
    while len(pts) > center_num:
        best_obj = float('inf')
        best_pt = 0
        for i, pt in enumerate(pts):
            pts = np.delete(pts, i, axis=0)
            obj = center_selection_dist(pts_old, pts)
            if obj < best_obj:
                best_obj = obj
                best_pt = i
            pts = np.insert(pts, i, pt, axis=0)
        pts = np.delete(pts, best_pt, axis=0)
    return center_selection_dist(pts_old, pts), pts


def random_points() -> Tuple[np.ndarray, int]:
    """
    Generate random points and the number of centers.

    Returns
    -------
    Tuple[np.ndarray, int]
        The points and the number of centers.
    """
    pts = np.random.rand(50, 2)
    center_num = 6
    return pts, center_num


def random_symmetry() -> Tuple[np.ndarray, int]:
    """
    Generate random points with symmetry and the number of centers.

    Returns
    -------
    Tuple[np.ndarray, int]
        The points and the number of centers.
    """
    pts = np.random.rand(8, 2)
    pts = np.concatenate((pts, pts[:, [1, 0]]))
    pts = np.concatenate((pts, np.column_stack((pts[:, 0], -pts[:, 1]))))
    pts = np.concatenate((pts, np.column_stack((-pts[:, 0], pts[:, 1]))))
    center_num = 8
    return pts, center_num


def random_clustered() -> Tuple[np.ndarray, int]:
    """
    Generate random points with clusters and the number of centers.

    Returns
    -------
    Tuple[np.ndarray, int]
        The points and the number of centers.
    """
    center_num = 5
    n_pts_per_cluster = 8
    cluster_centers = np.random.rand(center_num, 2) * 7
    pts = cluster_centers
    for i in range(center_num):
        pts = np.concatenate((pts, np.random.rand(n_pts_per_cluster, 2) + cluster_centers[i]))

    return pts, center_num


def multiple_layer_circle() -> Tuple[np.ndarray, int]:
    """
    Generate points on multiple layers of circles and the number of centers.

    Returns
    -------
    Tuple[np.ndarray, int]
        The points and the number of centers.
    """
    center = (0, 0)
    pts = np.array(center).reshape(1, 2)
    pts = np.concatenate((pts, get_circle_pts(center, 0.5, 6)))
    pts = np.concatenate((pts, get_circle_pts(center, 1, 12)))
    pts = np.concatenate((pts, get_circle_pts(center, 1.5, 18)))
    pts = np.concatenate((pts, get_circle_pts(center, 2, 24)))
    center_num = 10
    return pts, center_num


def task2_simple() -> Tuple[np.ndarray, int, np.ndarray]:
    """
    Simple example for task 2.
    Returns
    -------
    Tuple[np.ndarray, int]
        The points and the number of centers.
    """
    pts = get_circle_pts((0, 0), 2, 8)
    pts = np.delete(pts, 1, axis=0)
    pts = np.delete(pts, 2, axis=0)
    center_num = 1
    opt_max = np.array([[0, 0]])
    return pts, center_num, opt_max


def task2_hard() -> Tuple[np.ndarray, int, np.ndarray]:
    """
    Simple example for task 2.
    Returns
    -------
    Tuple[np.ndarray, int]
        The points and the number of centers.
    """
    pts = get_circle_pts((0, 0), 2, 8)
    pts = np.delete(pts, 1, axis=0)
    pts = np.delete(pts, 2, axis=0)
    pts = np.concatenate((pts, np.random.rand(6, 2)))
    center_num = 1
    opt_max = np.array([[0, 0]])
    return pts, center_num, opt_max


def main_random(plot: bool = True,
                n_epoch: Tuple[int, int | None] | None = (1000, 100),
                brutal_force: bool | Literal['auto'] = False,
                brutal_force_limit: int = 10_000_000) -> None:
    if plot:
        pts, center_num = random_clustered()
        if brutal_force == 'auto':
            brutal_force = brutal_force_limit >= comb(len(pts), center_num)
        dist, centers = dist_based_greedy_inclusion(pts, center_num)
        print(f'Distance-based greedy inclusion: {dist}')
        draw_points(pts, centers)
        dist, centers = dist_based_greedy_exclusion(pts, center_num)
        print(f'Distance-based greedy exclusion: {dist}')
        draw_points(pts, centers)
        dist, centers = obj_based_greedy_inclusion(pts, center_num)
        print(f'Objective-based greedy inclusion: {dist}')
        draw_points(pts, centers)
        dist, centers = obj_based_greedy_exclusion(pts, center_num)
        print(f'Objective-based greedy exclusion: {dist}')
        draw_points(pts, centers)
        if brutal_force:
            center_selection_fn = center_selection_int if pts.dtype == np.int32 else center_selection_float
            dist, centers = center_selection_fn(pts, center_num)
            print(f'Brutal force: {dist}')
            draw_points(pts, centers)

    else:
        dists = [[], [], [], []]

        def print_fn():
            print(f'Distance-based greedy inclusion: {np.mean(dists[0])}, {np.std(dists[0])}')
            print(f'Distance-based greedy exclusion: {np.mean(dists[1])}, {np.std(dists[1])}')
            print(f'Objective-based greedy inclusion: {np.mean(dists[2])}, {np.std(dists[2])}')
            print(f'Objective-based greedy exclusion: {np.mean(dists[3])}, {np.std(dists[3])}')

        for e in range(n_epoch[0]):
            pts, center_num = random_points()
            dist, _ = dist_based_greedy_inclusion(pts, center_num)
            dists[0].append(dist)
            dist, _ = dist_based_greedy_exclusion(pts, center_num)
            dists[1].append(dist)
            dist, _ = obj_based_greedy_inclusion(pts, center_num)
            dists[2].append(dist)
            dist, _ = obj_based_greedy_exclusion(pts, center_num)
            dists[3].append(dist)
            if n_epoch[1] is not None and (e + 1) % n_epoch[1] == 0:
                print(f'Epoch {e}')
                print_fn()
        print_fn()


def main_make_gif():
    pts, center_num = random_points()
    figs = []
    pts_old, pts = pts, deepcopy(pts)
    while len(pts) > center_num:
        best_obj = float('inf')
        best_pt = 0
        for i, pt in enumerate(pts):
            pts = np.delete(pts, i, axis=0)
            obj = center_selection_dist(pts_old, pts)
            if obj < best_obj:
                best_obj = obj
                best_pt = i
            pts = np.insert(pts, i, pt, axis=0)
        pts = np.delete(pts, best_pt, axis=0)
        fig = plt.figure()
        axis = fig.gca()
        draw_points(pts_old, pts, fig=axis, show=False)
        figs.append(fig)
    create_animated_gif(figs, 'obj_based_greedy_exclusion.gif', duration=400)


# noinspection PyTypeChecker
def pos_task2_cost(centers: np.ndarray | List[List[int | float]], pts: np.ndarray | List[List[int | float]], metric: Literal['se', 'el']) -> float:
    """
    Calculate pos fitness

    Parameters
    ----------
    centers : np.ndarray | List[List[int | float]]
        The centers.
    pts : np.ndarray | List[List[int | float]]
        The points.
    metric : Literal['se', 'el']
        The dist metric to use. 'se' for squared euclidean distance, 'el' for euclidean distance.

    Returns
    -------
    float
        The cost.
    """
    pts = np.array(pts)
    centers = np.array(centers)
    if metric == 'el':
        return np.sum(np.min(np.linalg.norm(pts[:, np.newaxis] - centers, axis=2), axis=1))
    elif metric == 'se':
        return np.sum(np.min(np.linalg.norm(pts[:, np.newaxis] - centers, axis=2), axis=1) ** 2)
    else:
        raise ValueError(f'Unknown metrix: {metric}')


def pos_task2_opt(pts: np.ndarray, center_num: int, init: np.ndarray | None, metric: Literal['se', 'el']) -> Tuple[float, np.ndarray]:
    if init is None:
        init = np.random.choice(pts.shape[0], center_num, replace=False)
        init = pts[init]
    min_x, max_x = np.min(pts[:, 0]), np.max(pts[:, 0])
    min_y, max_y = np.min(pts[:, 1]), np.max(pts[:, 1])
    bounds = [(min_x, max_x), (min_y, max_y)] * center_num
    init = init.flatten()
    cost_fn = partial(pos_task2_cost, pts=pts, metric=metric)
    dist, flat_centers = simple_pso.minimize_custom(cost_fn, [init], bounds, num_particles=50, maxiter=1000, verbose=False)
    centers = np.array(flat_centers).reshape(center_num, 2)
    return dist, centers


def main_task2_clustering():
    pts, center_num, opt_max = task2_simple()
    print(f'Optimal MaxDist: {center_selection_dist(pts, opt_max)}')
    print(f'Optimal MaxDist Center: {opt_max}')
    draw_points(pts, opt_max)
    init_centers = opt_max
    dist, centers = pos_task2_opt(pts, center_num, init_centers, 'se')
    print(f'Squared Euclidean: {dist}')
    print(f'Squared Euclidean Center: {centers}')
    draw_points(pts, centers)
    dist, centers = pos_task2_opt(pts, center_num, init_centers, 'el')
    print(f'Euclidean: {dist}')
    print(f'Euclidean Center: {centers}')
    draw_points(pts, centers)


if __name__ == '__main__':
    # main_random(plot=False, n_epoch=(100, None), brutal_force='auto')
    # main_make_gif()
    main_task2_clustering()
