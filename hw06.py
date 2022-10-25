from copy import copy, deepcopy
from typing import Tuple, List

import numpy as np

from hw05 import center_selection_dist, draw_points
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
    center_num = 2
    return pts, center_num


def main_random(plot: bool = True, n_epoch: Tuple[int, int | None] | None = (1000, 100)) -> None:
    if plot:
        pts, center_num = random_points()
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


if __name__ == '__main__':
    main_random(plot=False, n_epoch=(100, None))
