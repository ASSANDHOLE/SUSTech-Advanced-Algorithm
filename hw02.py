import itertools
import math
from typing import Tuple, List, Any, Optional

import numpy as np

import matplotlib.pyplot as plt
from numpy import ndarray

from algorithms.algorithms_impl_wrapper import tsp_naive as fast_tsp_naive


def travelling_salesman_naive(pts: np.ndarray) -> Tuple[List[np.ndarray], float]:
    """
    Parameters
    ----------
    pts : np.ndarray
        Points to visit.

    Returns
    -------
    Tuple[List[np.ndarray], float]
        The path and the total distance.
    """
    best_paths = []
    best_dist = float('inf')
    for path in itertools.permutations(pts):
        dist = 0
        for i in range(len(path) - 1):
            dist += np.linalg.norm(path[i] - path[i + 1])
        dist += np.linalg.norm(path[-1] - path[0])
        if dist < best_dist:
            best_dist = dist
            best_paths = [path]
        elif dist == best_dist:
            best_paths.append(path)
    return best_paths, best_dist


def travelling_salesman_greedy(pts: np.ndarray, start_point: Optional[int] = None):
    """
    Parameters
    ----------
    pts : np.ndarray
        Points to visit.
    start_point : Optional[int]
        Start point. None for all possible start points.

    Returns
    -------
    Tuple[Tuple[best_path_idx, path, float], Tuple[worst_path_idx, path, float]]
        The path and the total distance.
    """

    def alternative_in(collection: List[np.ndarray], item: np.ndarray) -> bool:
        for it in collection:
            if np.array_equal(it, item):
                return True
        return False

    def alternative_index(collection: List[np.ndarray], item: np.ndarray) -> int:
        for it in range(len(collection)):
            if np.array_equal(collection[it], item):
                return it
        return -1

    best_paths = []
    best_dist = float('inf')
    worst_paths = []
    worst_dist = float('-inf')
    possible_starts = range(len(pts)) if start_point is None else [start_point]
    for start_point in possible_starts:
        path = [pts[start_point]]
        dist = 0
        for i in range(len(pts) - 1):
            next_pt = None
            next_dist = float('inf')
            for pt in pts:
                if not alternative_in(path, pt):
                    d = np.linalg.norm(path[-1] - pt)
                    if d < next_dist:
                        next_dist = d
                        next_pt = pt
            path.append(next_pt)
            dist += next_dist
        dist += np.linalg.norm(path[-1] - path[0])
        if dist < best_dist:
            best_dist = dist
            best_paths = [path]
        elif dist == best_dist:
            best_paths.append(path)

        if dist > worst_dist:
            worst_dist = dist
            worst_paths = [path]
        elif dist == worst_dist:
            worst_paths.append(path)
    best_paths_idx = []
    for path in best_paths:
        best_paths_idx.append([alternative_index(pts, pt) for pt in path])
    worst_paths_idx = []
    for path in worst_paths:
        worst_paths_idx.append([alternative_index(pts, pt) for pt in path])
    return (best_paths_idx, best_paths, best_dist), (worst_paths_idx, worst_paths, worst_dist)


def local_search_best_first(pts: np.ndarray, route: List[np.ndarray]) -> tuple[
    list[int], list[ndarray] | Any, float | Any]:
    """Local search.

    Parameters
    ----------
    pts : np.ndarray
        Points to visit.
    route : List[np.ndarray]
        Initial route.

    Returns
    -------
    Tuple[List[np.ndarray], float]
        The path and the total distance.
    """
    dist_between = np.zeros((len(pts), len(pts)))
    for i in range(len(pts)):
        for j in range(len(pts)):
            dist_between[i, j] = np.linalg.norm(pts[i] - pts[j])
    best_path = route.copy()

    def idx_of(item: np.ndarray) -> int:
        for _i in range(len(pts)):
            if np.array_equal(pts[_i], item):
                return _i
        return -1

    def get_dist(path: List[np.ndarray]) -> float:
        dist = 0
        for _i in range(len(path) - 1):
            dist += dist_between[idx_of(path[_i]), idx_of(path[_i + 1])]
        dist += dist_between[idx_of(path[-1]), idx_of(path[0])]
        return dist

    best_dist = get_dist(best_path)

    while True:
        flag = False
        local_best_path = best_path.copy()
        local_best_dist = best_dist
        for i in range(len(route)):
            for j in range(i + 1, len(route)):
                new_path = best_path.copy()
                new_path[i:j] = new_path[i:j][::-1]
                new_dist = get_dist(new_path)
                if new_dist < local_best_dist:
                    local_best_path = new_path
                    local_best_dist = new_dist
                    flag = True
        if flag:
            best_path = local_best_path
            best_dist = local_best_dist
        else:
            break
    best_path_idx = [idx_of(pt) for pt in best_path]
    return best_path_idx, best_path, best_dist


def get_circle_pts(center: Tuple[float, float], radius: float, n: int = 100, start: float = 0) -> np.ndarray:
    """Get points on a circle.

    Parameters
    ----------
    center : Tuple[float, float]
        Center of the circle.
    radius : float
        Radius of the circle.
    n : int, optional
        Number of points, by default 100
    start: float, optional
        Start angle, by default 0

    Returns
    -------
    np.ndarray
        Points on the circle.
    """
    pts = np.zeros((n, 2))
    for i in range(n):
        angle = 2 * math.pi * i / n + start
        pts[i, 0] = center[0] + radius * math.cos(angle)
        pts[i, 1] = center[1] + radius * math.sin(angle)
    return pts


def task1_1() -> np.ndarray:
    """Task 1-1.

    Returns
    -------
    np.ndarray
        Points on the circle.
    """
    return get_circle_pts((0, 0), 1, 10)


def task1_2() -> np.ndarray:
    """Task 1-2.

    Returns
    -------
    np.ndarray
        Points on the circle.
    """
    pts = get_circle_pts((0, 0), 1, 15)
    pts = np.concatenate((pts, np.zeros((1, 2))), axis=0)
    return pts


def task2_1() -> np.ndarray:
    pts = [[0, 0], [0, 1], [2, 1], [2, 0], [1, -2]]
    return np.array(pts)


def task2_2():
    pts = [[-1.45, 4.2], [3.33, 7.2], [2.104, 3.343], [5.234, -12], [-6, 3.33]]
    return np.array(pts)


def task3_1():
    pts1 = get_circle_pts((0, 0), 2.3, 5)
    pts2 = get_circle_pts((0, 0), 1.0, 5, 2 * math.pi / (5*2))
    pts = np.concatenate((pts1, pts2), axis=0)
    return pts


def random_n_pts(n: int) -> np.ndarray:
    """Generate n random points.

    Parameters
    ----------
    n : int
        Number of points.

    Returns
    -------
    np.ndarray
        Random points.
    """
    return np.random.rand(n, 2)


def draw_pts(pts: np.ndarray, color: str = 'k', s: Optional[float] = None, numbering: bool = False):
    """Draw points.

    Parameters
    ----------
    pts : np.ndarray
        Points to draw.
    color : str, optional
        Color of the points, by default "k"
    s : Optional[float], optional
        Size of the points, by default None
    numbering : bool
        Whether to number the points, by default False
    """
    fig, ax = plt.subplots()
    ax.scatter(pts[:, 0], pts[:, 1], color=color, s=s)
    ax.axis('equal')
    plt.show()
    if numbering:
        fig, ax = plt.subplots()
        ax.scatter(pts[:, 0], pts[:, 1], color=color, s=s)
        ax.axis('equal')
        for i in range(len(pts)):
            ax.annotate(str(i), (pts[i, 0], pts[i, 1]))
        plt.show()


def float_equal(a, b):
    return abs(a - b) < 1e-7


def main_opt():
    i = 0
    while True:
        i += 1
        if i % 100 == 0:
            print(i)
        pts = random_n_pts(10)
        # a = travelling_salesman_naive(pts)
        a = fast_tsp_naive(pts)
        res = travelling_salesman_greedy(pts)
        if float_equal(a[1], res[0][-1]):
            continue

        ll = local_search_best_first(pts, res[0][1][0])
        if float_equal(res[0][-1], ll[-1]):
            break

    print(a)
    print(res[0][0][0], res[0][-1])
    print(ll[0], ll[-1])
    draw_pts(pts, numbering=True)


def main_norm():
    pts = task2_1()
    # print(travelling_salesman_naive(pts))
    sol_greedy = travelling_salesman_greedy(pts)
    res = local_search_best_first(pts, sol_greedy[1][0][0])
    print(sol_greedy[0][1])
    print(sol_greedy[1])
    print(res)
    draw_pts(pts)


def test1():
    pts = [[0, 0.01*a] for a in range(101)]
    pts += [[0.01, 1], [0.02, 1], [0.04, 1.01]]
    pts += [[0.03, 1-0.01*a] for a in range(101)]
    pts += [[0.02, 0], [0.01, 0]]
    pts = np.array(pts)
    route_len = 0
    for i in range(len(pts)-1):
        route_len += np.linalg.norm(pts[i+1] - pts[i])
    route_len += np.linalg.norm(pts[0] - pts[-1])
    print(route_len)
    print(travelling_salesman_greedy(pts, 0)[1])
    draw_pts(pts)


def test2():
    pts = [[0, 0.01], [0, 0.02], [0, 0.03], [0, 0.04], [0, 0.05], [0, 0.06], [0, 0.07], [0, 0.08], [0, 0.09], [0, 0.1], [-0.01, 0.105] ,[0, 0.11], [0, 0.12], [0, 0.13], [0, 0.14], [0, 0.15], [0, 0.16], [0, 0.17], [0, 0.18], [0, 0.19], [0, 0.2], [0, 0.21], [0, 0.22], [0, 0.23], [0, 0.24], [0, 0.25], [0, 0.26], [0, 0.27], [0, 0.28], [0, 0.29], [0, 0.3], [-0.01, 0.305],[0, 0.31], [0, 0.32], [0, 0.33], [0, 0.34], [0, 0.35], [0, 0.36], [0, 0.37], [0, 0.38], [0, 0.39], [0, 0.4], [0, 0.41], [0, 0.42], [0, 0.43], [0, 0.44], [0, 0.45], [0, 0.46], [0, 0.47], [0, 0.48], [0, 0.49], [0, 0.5], [-0.01, 0.505],[0, 0.51], [0, 0.52], [0, 0.53], [0, 0.54], [0, 0.55], [0, 0.56], [0, 0.57], [0, 0.58], [0, 0.59], [0, 0.6], [0, 0.61], [0, 0.62], [0, 0.63], [0, 0.64], [0, 0.65], [0, 0.66], [0, 0.67], [0, 0.68], [0, 0.69], [0, 0.7], [-0.01, 0.705], [0, 0.71], [0, 0.72], [0, 0.73], [0, 0.74], [0, 0.75], [0, 0.76], [0, 0.77], [0, 0.78], [0, 0.79], [0, 0.8], [0, 0.81], [0, 0.82], [0, 0.83], [0, 0.84], [0, 0.85], [0, 0.86], [0, 0.87], [0, 0.88], [0, 0.89], [0, 0.9], [-0.01, 0.905],[0, 0.91], [0, 0.92], [0, 0.93], [0, 0.94], [0, 0.95], [0, 0.96], [0, 0.97], [0, 0.98], [0, 0.99], [0, 1]]
    pts += [[0.01, 1], [0.02, 1]]
    pts += [[0.03, 1.0], [0.03, 0.99], [0.03, 0.98], [0.03, 0.97], [0.03, 0.96], [0.03, 0.95], [0.03, 0.94], [0.03, 0.93], [0.03, 0.92], [0.03, 0.91], [0.03, 0.9], [0.03, 0.89], [0.03, 0.88], [0.03, 0.87], [0.03, 0.86], [0.03, 0.85], [0.03, 0.84], [0.03, 0.83], [0.03, 0.82], [0.03, 0.81], [0.04, 0.805],[0.03, 0.8], [0.03, 0.79], [0.03, 0.78], [0.03, 0.77], [0.03, 0.76], [0.03, 0.75], [0.03, 0.74], [0.03, 0.73], [0.03, 0.72], [0.03, 0.71], [0.03, 0.7], [0.03, 0.69], [0.03, 0.68], [0.03, 0.67], [0.03, 0.66], [0.03, 0.65], [0.03, 0.64], [0.03, 0.63], [0.03, 0.62], [0.03, 0.61], [0.04, 0.605],[0.03, 0.6], [0.03, 0.59], [0.03, 0.58], [0.03, 0.57], [0.03, 0.56], [0.03, 0.55], [0.03, 0.54], [0.03, 0.53], [0.03, 0.52], [0.03, 0.51], [0.03, 0.5], [0.03, 0.49], [0.03, 0.48], [0.03, 0.47], [0.03, 0.456], [0.03, 0.45], [0.03, 0.44], [0.03, 0.43], [0.03, 0.42], [0.03, 0.41], [0.04, 0.405],[0.03, 0.4], [0.03, 0.39], [0.03, 0.38], [0.03, 0.37], [0.03, 0.36], [0.03, 0.35], [0.03, 0.34], [0.03, 0.33], [0.03, 0.32], [0.03, 0.31], [0.03, 0.30], [0.03, 0.29], [0.03, 0.28], [0.03, 0.27], [0.03, 0.26], [0.03, 0.25], [0.03, 0.24], [0.03, 0.23], [0.03, 0.22], [0.03, 0.21], [0.04, 0.205], [0.03, 0.2], [0.03, 0.19], [0.03, 0.18], [0.03, 0.17], [0.03, 0.16], [0.03, 0.15], [0.03, 0.14], [0.03, 0.13], [0.03, 0.12], [0.03, 0.11], [0.03, 0.1], [0.03, 0.09], [0.03, 0.08], [0.03, 0.07], [0.03, 0.06], [0.03, 0.05], [0.03, 0.04], [0.03, 0.03], [0.03, 0.02], [0.03, 0.01], [0.03, 0.0]]
    pts += [[0.02, 0], [0.01, 0]]
    pts = np.array(pts)
    route_len = 0
    for i in range(len(pts)-1):
        route_len += np.linalg.norm(pts[i+1] - pts[i])
    route_len += np.linalg.norm(pts[0] - pts[-1])
    print(route_len)
    print(travelling_salesman_greedy(pts)[1])


if __name__ == "__main__":
    main_opt()

