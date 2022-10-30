from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

from algorithms import kmeans_2d_int, kmeans_2d_float
from algorithms import kmedoids_2d_int, kmedoids_2d_float
from algorithms import fuzzy_c_means_2d_int, fuzzy_c_means_2d_float
from algorithms import square_euclidian_dist

from hw01 import get_circle_pts
from hw05 import task1_1_1 as hw05_task1_1_1


def generate_normal_dist_pts(center: np.ndarray, cov: np.ndarray, n: int) -> np.ndarray:
    if not n > 0:
        return np.array([[]])
    arr = np.random.multivariate_normal(center, cov, n - 1)
    # add a point to make the center of the cluster be the center of the distribution
    new_center = np.mean(arr, axis=0)
    new_pt = center + (center - new_center) * (n - 1)
    return np.concatenate((arr, [new_pt]))


def plot_cluster(points: np.ndarray, centers: np.ndarray, labels: np.ndarray):
    plt.scatter(points[:, 0], points[:, 1], c=labels)
    plt.scatter(centers[:, 0], centers[:, 1], c='black', marker='x', linewidths=3)
    plt.axis('equal')
    plt.show()


def plot_fuzzy_cluster(points: np.ndarray, centers: np.ndarray, membership: np.ndarray, m: float | None = None):
    n_colors = centers.shape[0]

    # use HSV color space to easily generate diverse colors
    def hsv_to_rgb(h: float, s: float, v: float) -> Tuple[float, float, float]:
        h_i = int(h * 6)
        f = h * 6 - h_i
        p = v * (1 - s)
        q = v * (1 - f * s)
        t = v * (1 - (1 - f) * s)
        if h_i == 0:
            return v, t, p
        elif h_i == 1:
            return q, v, p
        elif h_i == 2:
            return p, v, t
        elif h_i == 3:
            return p, q, v
        elif h_i == 4:
            return t, p, v
        else:
            return v, p, q

    hsv_colors = np.linspace(0, 1, n_colors, endpoint=False)
    rgb_colors = np.array([hsv_to_rgb(h, 1, 1) for h in hsv_colors])
    colors = []
    for i in range(points.shape[0]):
        color = np.zeros(3)
        for j in range(len(centers)):
            color += [membership[i, j] * c for c in rgb_colors[j]]
        colors.append(color)

    pts_size = plt.rcParams['lines.markersize'] ** 2
    center_size = pts_size * 2

    plt.scatter(points[:, 0], points[:, 1], c=colors, s=pts_size)
    plt.scatter(centers[:, 0], centers[:, 1], c=rgb_colors,
                s=center_size, marker='x', linewidths=3)
    plt.axis('equal')
    if m is not None:
        plt.title(f'Fuzzy C-Means {m=:.2f}')
    plt.show()


def task2() -> Tuple[np.ndarray, int]:
    """
    Task 2: Demonstrate the difference between K-Means and K-Medoids.

    Returns
    -------
    Tuple[np.ndarray, int]
        Points and k.
    """
    pts, *_ = hw05_task1_1_1()
    k = 2
    return pts, k


def task3() -> Tuple[np.ndarray, int, np.ndarray, np.ndarray]:
    """
    Task 3: Demonstrate the importance of m in Fuzzy C-Means.

    Returns
    -------
    Tuple[np.ndarray, int, np.ndarray]
        Points and k,
         True centers and membership matrix.
    """
    centers = np.array([
        [0, 0],
        [2, 6],
        [-2, 4]
    ])
    dim = [(1, 6), (2, 10), (2.7, 14)]
    n_random = 0
    cov = np.diag([.8, .8])
    pts = get_circle_pts((0, 0), *dim[0])
    for d in dim[1:]:
        pts = np.concatenate((pts, get_circle_pts((0, 0), *d)))
    pts = np.concatenate([np.concatenate(
        (pts, generate_normal_dist_pts(np.zeros(2), cov, n_random))
    ) if n_random > 0 else pts + c for c in centers])
    k = 3
    opt_centers = centers
    opt_membership = np.zeros((pts.shape[0], k))
    group_num = sum([d[1] for d in dim]) + n_random
    for i in range(k):
        opt_membership[i * group_num: (i + 1) * group_num, i] = 1
    return pts, k, opt_centers, opt_membership


def main_task1():
    random_init = [[], [], []]
    kmeans_pp = [[], [], []]
    n_runs = 1000
    n_pts = 1000
    k = 20
    eps = 1e-6
    max_iter = 1000
    for i in range(n_runs):
        pts = np.random.rand(n_pts, 2)
        centers, labels, n_iter, converged = kmeans_2d_float(pts, k, 'random', eps=eps, max_iter=max_iter)
        dist = np.sum(square_euclidian_dist(pts, centers, labels))
        random_init[0].append(dist)
        random_init[1].append(n_iter)
        random_init[2].append(converged)
        centers, labels, n_iter, converged = kmeans_2d_float(pts, k, 'kmeans++', eps=eps, max_iter=max_iter)
        dist = np.sum(square_euclidian_dist(pts, centers, labels))
        kmeans_pp[0].append(dist)
        kmeans_pp[1].append(n_iter)
        kmeans_pp[2].append(converged)
    print('random init:')
    print(f'\tdist: {np.mean(random_init[0])} +- {np.std(random_init[0])}')
    print(f'\tn_iter: {np.mean(random_init[1])} +- {np.std(random_init[1])}')
    print(f'\tconverged_rate: {np.mean(random_init[2])}')
    print('\nkmeans++:')
    print(f'\tdist: {np.mean(kmeans_pp[0])} +- {np.std(kmeans_pp[0])}')
    print(f'\tn_iter: {np.mean(kmeans_pp[1])} +- {np.std(kmeans_pp[1])}')
    print(f'\tconverged_rate: {np.mean(kmeans_pp[2])}')


def main_task2():
    pts, k = task2()
    centers, labels, n_iter, converged = kmeans_2d_int(pts, k)
    dist = np.sum(square_euclidian_dist(pts, centers, labels))
    print(f'K-Means: {dist=}, {n_iter=}, {converged=}')
    plot_cluster(pts, centers, labels)
    centers_idx, labels, n_iter, converged = kmedoids_2d_int(pts, k)
    centers = pts[centers_idx]
    dist = np.sum(square_euclidian_dist(pts, centers, labels))
    print(f'K-Medoids: {dist=}, {n_iter=}, {converged=}')
    plot_cluster(pts, centers, labels)


def main_task3():
    pts, k, true_centers, true_mem = task3()
    plot_fuzzy_cluster(pts, true_centers, true_mem, m=None)
    arg_true_mem = np.argmax(true_mem, axis=1)

    centers, labels, n_iter, converged = kmeans_2d_float(pts, k)
    dist = np.sum(square_euclidian_dist(pts, centers, labels))
    kmeans_correct = 0
    for i in range(k):
        max_correct = 0
        for j in range(k):
            correct = np.sum(arg_true_mem[labels == i] == j)
            if correct > max_correct:
                max_correct = correct
        kmeans_correct += max_correct
    print(f'K-Means: {dist=}, {n_iter=}, {converged=}')
    print(f'\tcorrect: {kmeans_correct}/{pts.shape[0]}={kmeans_correct / pts.shape[0] * 100:.2f}%')
    plot_cluster(pts, centers, labels)

    def test_m(_m):
        centers, membership, n_iter, converged = fuzzy_c_means_2d_float(pts, k, _m, max_iter=1000)
        print(f'Fuzzy C-Means with m={_m}: {n_iter=}, {converged=}')
        correct = 0
        arg_mem = np.argmax(membership, axis=1)
        for i in range(pts.shape[0]):
            max_correct = 0
            for j in range(pts.shape[0]):
                crt = np.sum(((arg_true_mem == j) & (arg_mem == i)).astype(int))
                if crt > max_correct:
                    max_correct = crt
            correct += max_correct
        print(f'\tcorrect: {correct}/{pts.shape[0]}={correct / pts.shape[0] * 100:.2f}%')
        plot_fuzzy_cluster(pts, centers, membership, m=_m)

    ms = [1.00001, 1.1, 1.2, 1.3, 1.5, 1.7, 2, 3, 10]
    for m in ms:
        test_m(m)


if __name__ == '__main__':
    main_task3()
