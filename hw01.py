import itertools
import math
from typing import Tuple, List, Union, Any

import numpy as np

import matplotlib.pyplot as plt


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


def travelling_salesman_greedy(pts: np.ndarray) -> Tuple[List[List[Any]], float]:
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
    def alternative_in(collection: List[np.ndarray], item: np.ndarray) -> bool:
        for it in collection:
            if np.array_equal(it, item):
                return True
        return False
    best_paths = []
    best_dist = float('inf')
    for start in range(len(pts)):
        path = [pts[start]]
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

    return best_paths, best_dist


def get_circle_pts(center: Tuple[float, float], radius: float, n: int = 100) -> np.ndarray:
    """Get points on a circle.

    Parameters
    ----------
    center : Tuple[float, float]
        Center of the circle.
    radius : float
        Radius of the circle.
    n : int, optional
        Number of points, by default 100

    Returns
    -------
    np.ndarray
        Points on the circle.
    """
    pts = np.zeros((n, 2))
    for i in range(n):
        angle = 2 * math.pi * i / n
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
    pts = [[-7.5, 5.5], [-7.5, 3], [-7.5, 0], [-4, 0],
           [0, 0],
           [4, 0], [7.5, 0], [7.5, -3], [7.5, -5.5]]
    return np.array(pts)


def task2_2() -> np.ndarray:
    pts = [[-5, 0], [-4, -2], [-3, -3], [-2, -4], [0, -5],
           [2, -4], [3, -3], [4, -2], [5, 0],
           [4, 2], [3, 3], [2, 4], [0, 10],
           [-2, 4], [-3, 3], [-4, 2]]
    return np.array(pts)


def task3_1():
    pts = [[-9, 0], [-4, -2], [-3, -3], [-2, -4], [0, -5],
           [2, -4], [3, -3], [4, -2], [9, 0],
           [4, 2], [3, 3], [2, 4], [0, 5],
           [-2, 4], [-3, 3], [-4, 2]]
    return np.array(pts)


def task3_2():
    pts = [[-1.45, 4.2], [3.33, 7.2], [2.104, 3.343], [5.234, -12], [-6, 3.33]]
    return np.array(pts)


def task4_1():
    pts1 = get_circle_pts((0, 0), 2, 20)
    pts2 = get_circle_pts((0, 0), 1, 20)
    pts = np.concatenate((pts1, pts2), axis=0)
    return pts


def task4_2():
    pts = [[1, -2], [-.5, -1.5], [0.0, 0.0], [0.5, 0.5], [0.7, 0.7], [0.7, 0.3]]
    return np.array(pts)


def draw_pts(pts: np.ndarray, color: str = 'k'):
    """Draw points.

    Parameters
    ----------
    pts : np.ndarray
        Points to draw.
    color : str, optional
        Color of the points, by default "k"
    """
    plt.scatter(pts[:, 0], pts[:, 1], color=color)
    plt.axis('equal')
    plt.show()


def main():
    pts = task4_2()
    print(travelling_salesman_naive(pts))
    print(travelling_salesman_greedy(pts))
    draw_pts(pts)


if __name__ == "__main__":
    main()
