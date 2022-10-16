from typing import Tuple, List
from math import sqrt
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

import algorithms.simple_pso as pso
from algorithms import center_selection_dist, center_selection_int, center_selection_float

from hw01 import get_circle_pts


def greedy_center_selection(points: np.ndarray, center_num: int, init: int | None = None) -> Tuple[float, np.ndarray]:
    """
    Greedy center selection algorithm (2-approximation).

    Parameters
    ----------
    points : np.ndarray
        The points to be selected.
    center_num : int
        The number of centers to be selected.
    init : int | None
        The index of the initial center. If None, try all points as initial centers and return the best result.

    Returns
    -------
    Tuple[float, np.ndarray]
        The maximum distance to any center and the centers.
    """
    min_dist = np.inf
    if init is None:
        for i in range(points.shape[0]):
            dist, centers = greedy_center_selection(points, center_num, i)
            if dist < min_dist:
                min_dist = dist
                min_centers = centers
        return min_dist, min_centers
    centers = np.zeros((center_num, points.shape[1]), dtype=points.dtype)
    centers[0] = points[init]
    for i in range(1, center_num):
        dist = np.linalg.norm(points - centers[i - 1], axis=1)
        for j in range(i):
            dist = np.minimum(dist, np.linalg.norm(points - centers[j], axis=1))
        centers[i] = points[np.argmax(dist)]
    dist = center_selection_dist(points, centers)
    return dist, centers


def kmeans_get_centers(points: np.ndarray, k: int) -> np.ndarray:
    """
    Get the centers of k-means clustering.

    Parameters
    ----------
    points : np.ndarray
        The points to be clustered.
    k : int
        The number of clusters.

    Returns
    -------
    np.ndarray
        The centers.
    """
    kmeans = KMeans(n_clusters=k).fit(points)
    return kmeans.cluster_centers_


def pso_fitness(centers: np.ndarray, points: np.ndarray) -> float:
    if not isinstance(centers, np.ndarray):
        centers = np.array(centers)
    centers = centers.reshape((int(centers.shape[0] // 2), 2))
    return center_selection_dist(points, centers)


def pso_optimize(points: np.ndarray, center_num: int, init: List[np.ndarray], verbose: bool = True) -> np.ndarray:
    """
    Optimize the centers using particle swarm optimization.

    Parameters
    ----------
    points : np.ndarray
        The points to be selected.
    center_num : int
        The number of centers to be selected.
    init : List[np.ndarray
        The initial array of centers.
    verbose : bool
        Whether to print the result during optimization.

    Returns
    -------
    np.ndarray
        The optimized centers.
    """
    min_x, max_x = np.min(points, axis=0), np.max(points, axis=0)
    min_y, max_y = np.min(points, axis=0), np.max(points, axis=0)
    bound = [(min_x[0], max_x[0]), (min_y[1], max_y[1])] * center_num
    init = [i.flatten() for i in init]
    dist, centers = pso.minimize_custom(partial(pso_fitness, points=points), init, bound, num_particles=100, maxiter=200, verbose=verbose)
    return np.array(centers, dtype=np.float32).reshape((center_num, 2))


def draw_points(points: np.ndarray, centers: np.ndarray,
                points_color: str = 'b', center_color: str = 'r', off_pointer_color: str = '#00ff00') -> None:
    """
    Draw the points and centers.

    Parameters
    ----------
    points : np.ndarray
        The points to be drawn.
    centers : np.ndarray
        The centers to be drawn.
    points_color : str
        The color of the points.
    center_color : str
        The color of the centers that is on the point.
    off_pointer_color : str
        The color of the centers that is not the point.

    Returns
    -------
    None
    """
    def in_points(center: np.ndarray) -> bool:
        for i in range(points.shape[0]):
            if np.array_equal(center, points[i]):
                return True
        return False

    plt.scatter(points[:, 0], points[:, 1], c=points_color)
    on_point_centers = [center for center in centers if in_points(center)]
    off_point_centers = [center for center in centers if not in_points(center)]
    on_point_centers = np.array(on_point_centers)
    off_point_centers = np.array(off_point_centers)
    legend = ['Points']
    if on_point_centers.shape[0] > 0:
        plt.scatter(on_point_centers[:, 0], on_point_centers[:, 1], c=center_color)
        legend.append('On-point Centers')
    if off_point_centers.shape[0] > 0:
        plt.scatter(off_point_centers[:, 0], off_point_centers[:, 1], c=off_pointer_color)
        legend.append('Off-point Centers')
    plt.axis('equal')
    # plt.grid()
    plt.legend(legend)
    plt.show()


def task1_1_1() -> Tuple[np.ndarray, int, np.ndarray]:
    """
    Task 1.1.1

    Returns
    -------
    Tuple[np.ndarray, int, np.ndarray]
        The points and the center number. The last element is the optimal centers.
    """

    pts = np.array(get_circle_pts((0, 0), 1, 4), dtype=np.float32)
    cts = np.array([[0, 0], [6, 0], [3, 3*sqrt(3)]], dtype=np.float32)
    arr = [ct + pts for ct in cts]
    arr = np.concatenate(arr)
    opt_centers = cts
    return arr, 3, opt_centers


def task1_1_2() -> Tuple[np.ndarray, int, np.ndarray]:
    """
    Task 1.1.2

    Returns
    -------
    Tuple[np.ndarray, int, np.ndarray]
        The points and the center number. The last element is the optimal centers.
    """
    opt_centers = np.array([[0, 0]], dtype=np.float32)
    pts = np.array(get_circle_pts(opt_centers[0], 1, 20), dtype=np.float32)
    return pts, 1, opt_centers


def task1_2_1() -> Tuple[np.ndarray, int, np.ndarray]:
    """
    Task 1.2.1

    Returns
    -------
    Tuple[np.ndarray, int, np.ndarray]
        The points and the center number. The last element is the optimal centers.
    """
    points = np.array([[1, 1], [2, 2], [3, 3], [1, 3], [3, 1]])
    center_num = 1
    opt_centers = kmeans_get_centers(points, center_num)
    return points, center_num, opt_centers


def task1_2_2() -> Tuple[np.ndarray, int, np.ndarray]:
    """
    Task 1.2.2

    Returns
    -------
    Tuple[np.ndarray, int, np.ndarray]
        The points and the center number. The last element is the optimal centers.
    """
    points = np.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])
    center_num = 2
    opt_centers = np.array([points[1], points[3]])
    return points, center_num, opt_centers


def task1_random() -> np.ndarray:
    """
    Task 1. random generated points

    Returns
    -------
    np.ndarray
        The points.
    """
    points = np.random.rand(10, 2) * 10
    points = points.astype(np.float32)
    return points


def task2_1_1() -> Tuple[np.ndarray, int]:
    """
    Task 2.1.1

    Returns
    -------
    Tuple[np.ndarray, int]
        The points and the center number.
    """
    points = np.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]], dtype=np.int32)
    center_num = 2
    return points, center_num


def task2_1_2() -> None:
    """
    [[0.         0.        ]
     [0.         1.4098663 ]
     [0.         3.2946048 ]
     [0.         4.7801347 ]
     [0.79540265 0.        ]
     [1.0126642  4.015103  ]
     [0.88189405 9.458944  ]
     [1.8661066  4.278283  ]
     [2.3077295  5.483091  ]
     [3.7675345  1.7338527 ]
     [3.1150482  4.674516  ]
     [2.8992035  9.199391  ]
     [4.71731    0.        ]
     [4.47005    2.0162218 ]
     [4.0396733  9.308292  ]
     [4.0710897  8.280334  ]
     [3.9214568  8.978085  ]
     [4.903638   0.02900437]
     [4.185843   4.030243  ]
     [4.930054   9.08793   ]]
    ratio = 1.9986048936843872
    """
    pass


def task2_2() -> None:
    """
    can easily be acquired by random
    """
    pass


def task2_random_int() -> Tuple[np.ndarray, int]:
    """
    Task 2. random generated points

    Returns
    -------
    Tuple[np.ndarray, int]
        The points and the center number.
    """
    points = np.random.randint(0, 10, (10, 2), dtype=np.int32)
    # no identical points
    points = np.unique(points, axis=0)
    points = points.astype(np.int32)
    if points.shape[0] > 8:
        points = points[:8]
    elif points.shape[0] < 8:
        return task2_random_int()
    center_num = 3
    return points, center_num


def task3_init_select(points: np.ndarray, center_num: int | None) -> int:
    """
    Task 3

    Parameters
    ----------
    points : np.ndarray
        The points.
    center_num : int | None
        The center number. If None, will be treat as 1.

    Returns
    -------
    int
        The initial center idx.
    """
    center_num = 1 if center_num is None else center_num
    kmeans = KMeans(n_clusters=center_num).fit(points)
    label_count = np.bincount(kmeans.labels_)
    best_center = kmeans.cluster_centers_[np.argmax(label_count)]
    closest, _ = pairwise_distances_argmin_min(best_center.reshape(1, -1), points)
    return closest[0]


def task3_random() -> Tuple[np.ndarray, int]:
    """
    Task 3. random generated points

    Returns
    -------
    Tuple[np.ndarray, int]
        The points and the center number.
    """
    points = np.random.rand(20, 2) * 10
    points = points.astype(np.float32)
    center_num = 4
    return points, center_num


def main_task1():
    points, center_num, opt_centers = task1_2_2()
    dist, centers = greedy_center_selection(points, center_num)
    print(f'Greedy: {dist:.2f}')
    draw_points(points, centers)
    print(f'Optimal: {center_selection_dist(points, opt_centers):.2f}')
    draw_points(points, opt_centers)


def main_task1_pso_single():
    points = task1_random()
    center_num = 3
    init = kmeans_get_centers(points, center_num)
    greedy_dist, greedy_centers = greedy_center_selection(points, center_num)
    centers = pso_optimize(points, center_num, [init, greedy_centers], verbose=True)
    print(f'PSO: {center_selection_dist(points, centers):.2f}')
    draw_points(points, centers)
    print(f'Greedy: {greedy_dist:.2f}')
    draw_points(points, greedy_centers)


def main_task1_pso_all(epoch=200):
    center_num = 3
    best_two = [(np.inf, None), (np.inf, None)]
    worst_one = (0.0, None)
    for i in range(epoch):
        print(f'Epoch {i + 1}/{epoch}')
        points = task1_random()
        init = kmeans_get_centers(points, center_num)
        init = [init]
        greedy_dist = np.inf
        greedy_centers = None
        for e in range(points.shape[0]):
            gd, gc = greedy_center_selection(points, center_num, e)
            init.append(gc)
            if gd < greedy_dist:
                greedy_dist = gd
                greedy_centers = gc
        pso_centers = pso_optimize(points, center_num, init, verbose=False)
        pso_dist = center_selection_dist(points, pso_centers)
        ratio = greedy_dist / pso_dist
        if ratio < best_two[-1][0]:
            best_two[-1] = (ratio, (points, pso_centers, greedy_centers))
            best_two.sort()
        if ratio > worst_one[0]:
            worst_one = (ratio, (points, pso_centers, greedy_centers))

    print(f'good1: {best_two[0][0]:.2f}')
    print(f'greedy: {center_selection_dist(best_two[0][1][0], best_two[0][1][2]):.2f}')
    print(f'pso: {center_selection_dist(best_two[0][1][0], best_two[0][1][1]):.2f}')
    draw_points(best_two[0][1][0], best_two[0][1][1])
    draw_points(best_two[0][1][0], best_two[0][1][2])
    print(f'good2: {best_two[1][0]:.2f}')
    print(f'greedy: {center_selection_dist(best_two[1][1][0], best_two[1][1][2]):.2f}')
    print(f'pso: {center_selection_dist(best_two[1][1][0], best_two[1][1][1]):.2f}')
    draw_points(best_two[1][1][0], best_two[1][1][1])
    draw_points(best_two[1][1][0], best_two[1][1][2])
    print(f'bad: {worst_one[0]:.2f}')
    print(f'greedy: {center_selection_dist(worst_one[1][0], worst_one[1][2]):.2f}')
    print(f'pso: {center_selection_dist(worst_one[1][0], worst_one[1][1]):.2f}')
    draw_points(worst_one[1][0], worst_one[1][1])
    draw_points(worst_one[1][0], worst_one[1][2])


def main_task2():
    points, center_num = task2_1_1()
    center_selection_fn = center_selection_int if points.dtype == np.int32 else center_selection_float
    g_dist, centers = greedy_center_selection(points, center_num)
    print(f'Greedy: {g_dist:.2f}')
    draw_points(points, centers)
    dist, centers = center_selection_fn(points, center_num)
    print(f'PSO: {center_selection_dist(points, centers):.2f}')
    draw_points(points, centers)
    print(f'Ratio: {g_dist / dist:.2f}')


def task2_pso_fitness(points: np.ndarray, center_num: int) -> float:
    """
    Task 2. fitness function for PSO

    Parameters
    ----------
    points : np.ndarray
        The points.
    center_num : int
        The center number.

    Returns
    -------
    float
        The fitness value.
    """
    points = np.array(points, dtype=np.float32).reshape(-1, 2)
    g_dist, _ = greedy_center_selection(points, center_num)
    dist, _ = center_selection_float(points, center_num)
    ratio = g_dist / dist
    return -ratio


def main_task2_random(epoch=200):
    best_two = [(np.inf, None), (np.inf, None)]
    center_num = 0
    for i in range(epoch):
        print(f'Epoch {i + 1}/{epoch}')
        points, center_num = task2_random_int()
        g_dist, g_centers = greedy_center_selection(points, center_num)
        dist, centers = center_selection_int(points, center_num)
        ratio = g_dist / dist
        if all([b[0] == 1.0 for b in best_two]):
            break
        if ratio < best_two[-1][0]:
            best_two[-1] = (ratio, points)
            best_two.sort(key=lambda x: x[0])

    print(f'good1: {best_two[0][0]:.2f}')
    g_dist, g_centers = greedy_center_selection(best_two[0][1], center_num)
    dist, centers = center_selection_int(best_two[0][1], center_num)
    print(f'greedy: {g_dist:.2f}')
    draw_points(best_two[0][1], g_centers)
    print(f'opt: {dist:.2f}')
    draw_points(best_two[0][1], centers)
    print(f'good2: {best_two[1][0]:.2f}')
    g_dist, g_centers = greedy_center_selection(best_two[1][1], center_num)
    dist, centers = center_selection_int(best_two[1][1], center_num)
    print(f'greedy: {g_dist:.2f}')
    draw_points(best_two[1][1], g_centers)
    print(f'opt: {dist:.2f}')
    draw_points(best_two[1][1], centers)


def main_task2_rand_pso(epoch=200):
    n = 50
    rand_fn = task2_random_int
    center_selection_fn = center_selection_int
    worst_n = [(0.0, None) for _ in range(n)]
    pt_num = 0
    center_num = 0
    for i in range(epoch):
        print(f'Epoch {i + 1}/{epoch}')
        points, center_num = rand_fn()
        pt_num = points.shape[0]
        g_dist, g_centers = greedy_center_selection(points, center_num)
        dist, centers = center_selection_fn(points, center_num)
        ratio = g_dist / dist
        if ratio > worst_n[0][0]:
            worst_n[0] = (ratio, points)
            worst_n.sort(key=lambda x: x[0])
    init = [w[1].flatten() for w in worst_n]
    bound = [(0, 10)] * (pt_num * 2)
    dist, pts = pso.minimize_custom(partial(task2_pso_fitness, center_num=center_num), init, bound, num_particles=100, maxiter=500, verbose=True)
    pts = np.array(pts, dtype=np.float32).reshape(-1, 2)
    g_dist, g_centers = greedy_center_selection(pts, center_num)
    dist, centers = center_selection_float(pts, center_num)
    print(f'Greedy: {g_dist:.2f}')
    draw_points(pts, g_centers)
    print(f'PSO: {center_selection_dist(pts, centers):.2f}')
    draw_points(pts, centers)
    print(pts)


def main_task3(n_veal=200):
    ratio_best = []
    ratio_worst = []
    random_ratio_best = []
    random_ratio_worst = []
    for i in range(n_veal):
        print(f'Epoch {i + 1}/{n_veal}')
        points, center_num = task3_random()
        init_idx = task3_init_select(points, center_num)
        init_dist, _ = greedy_center_selection(points, center_num, init_idx)
        random_init_idx = np.random.randint(0, points.shape[0])
        random_init_dist, _ = greedy_center_selection(points, center_num, random_init_idx)
        worst_dist = 0
        best_dist = np.inf
        for j in range(points.shape[0]):
            dist, _ = greedy_center_selection(points, center_num, j)
            if dist > worst_dist:
                worst_dist = dist
            if dist < best_dist:
                best_dist = dist
        ratio_best.append(init_dist / best_dist)
        ratio_worst.append(init_dist / worst_dist)
        random_ratio_best.append(random_init_dist / best_dist)
        random_ratio_worst.append(random_init_dist / worst_dist)

    print(f'Best: {np.mean(ratio_best):.2f} ± {np.std(ratio_best):.2f}')
    print(f'Worst: {np.mean(ratio_worst):.2f} ± {np.std(ratio_worst):.2f}')
    print(f'Random Best: {np.mean(random_ratio_best):.2f} ± {np.std(random_ratio_best):.2f}')
    print(f'Random Worst: {np.mean(random_ratio_worst):.2f} ± {np.std(random_ratio_worst):.2f}')


if __name__ == '__main__':
    main_task3(1000)
