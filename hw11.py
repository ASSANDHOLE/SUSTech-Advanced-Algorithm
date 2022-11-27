import sys
import os
import ctypes
import subprocess
from typing import List, Tuple

import numpy as np

from hw08 import set_cover_greedy
from hw09 import graph_set_convert, iterate_graph
from hw09 import draw as draw_graph

_dll = None


def _dependency_check():
    global _dll
    if sys.platform == 'win32':
        if not os.path.exists('hw11_cpp/libhw11_cpp.dll'):
            raise RuntimeError('Cannot find libhw11_cpp.dll,'
                               ' build it first in the '
                               '"hw11_cpp/libhw11_cpp.dll".')
        _dll = ctypes.cdll.LoadLibrary('hw11_cpp/libhw11_cpp.dll')
    elif sys.platform == 'linux':
        if not os.path.exists('hw11_cpp/libhw11_cpp.so'):
            full_path = os.path.abspath(os.path.dirname(__file__))
            full_path = os.path.join(full_path, 'hw11_cpp', 'build.sh')
            cwd = os.getcwd()
            os.chdir(os.path.dirname(full_path))
            return_code = subprocess.run(['bash', os.path.basename(full_path)])
            os.chdir(cwd)
            if return_code.returncode != 0:
                raise RuntimeError('Cannot find libhw11_cpp.so,'
                                   ' try build by build.sh failed, '
                                   ' build it manually in the "hw11_cpp"')
        _dll = ctypes.cdll.LoadLibrary('hw11_cpp/libhw11_cpp.so')
    else:
        raise RuntimeError('Unregistered platform: {}'.format(sys.platform))


try:
    _dependency_check()
except OSError as e:
    print('Cannot load the library, please check the dependency.')
    print(e)
    print('possible fix for libstdc++ version `GLIBCXX...` not found in conda envs:')
    print('ln -s /usr/lib/x86_64-linux-gnu/libstdc++.so.6 /your/conda_dir/lib/libstdc++.so.6')
    sys.exit(1)


def _load_mip_solver():
    ndptr = np.ctypeslib.ndpointer
    mip_solver = _dll.Hw11MipSolver
    mip_solver.restype = ctypes.c_double
    mip_solver.argtypes = [
        ndptr(dtype=np.uint8, flags='aligned, c_contiguous'),
        ctypes.c_int, ctypes.c_int,
        ndptr(dtype=np.float32, flags='aligned, c_contiguous'),
        ndptr(dtype=np.int32, flags='aligned, c_contiguous'),
    ]
    return mip_solver


def _load_lp_based_solver():
    ndptr = np.ctypeslib.ndpointer
    lp_based_solver = _dll.Hw11LpBasedSolver
    lp_based_solver.restype = ctypes.c_double
    lp_based_solver.argtypes = [
        ndptr(dtype=np.uint8, flags='aligned, c_contiguous'),
        ctypes.c_int, ctypes.c_int,
        ndptr(dtype=np.float32, flags='aligned, c_contiguous'),
        ndptr(dtype=np.float32, flags='aligned, c_contiguous'),
    ]
    return lp_based_solver


_mip_solver = _load_mip_solver()
_lp_based_solver = _load_lp_based_solver()


def ip_solve(edges: List[List[int]] | np.ndarray,
             weights: List[int | float] | np.ndarray) -> \
        Tuple[np.ndarray, float]:
    """
    The Integer Programming solver for the problem.

    Parameters
    ----------
    edges : List[List[int]] | np.ndarray
        The graph data. [[node1, node2], ...]
    weights : List[int | float] | np.ndarray
        The weights of the nodes. [weight1, weight2, ...]

    Returns
    -------
    Tuple[np.ndarray, float]
        The index of the graph cover and the weight.
    """
    edge_mat = np.zeros((len(edges), len(weights)), dtype=np.uint8)
    for i, edge in enumerate(edges):
        edge_mat[i, edge[0]] = 1
        edge_mat[i, edge[1]] = 1
    weights = np.array(weights, dtype=np.float32)
    cover = np.zeros(len(weights), dtype=np.int32)
    weight = _mip_solver(edge_mat, len(weights), len(edges), weights, cover)
    cover = np.where(cover == 1)[0]
    return cover, weight


def lp_based_solve(edges: List[List[int]] | np.ndarray,
                   weights: List[int | float] | np.ndarray) -> \
        Tuple[np.ndarray, float]:
    """
    The Linear Programming based solver for the problem.

    Parameters
    ----------
    edges : List[List[int]] | np.ndarray
        The graph data. [[node1, node2], ...]
    weights : List[int | float] | np.ndarray
        The weights of the nodes. [weight1, weight2, ...]

    Returns
    -------
    Tuple[np.ndarray, float]
        The index of the graph cover and the weight.
    """
    edge_mat = np.zeros((len(edges), len(weights)), dtype=np.uint8)
    for i, edge in enumerate(edges):
        edge_mat[i, edge[0]] = 1
        edge_mat[i, edge[1]] = 1
    weights = np.array(weights, dtype=np.float32)
    cover = np.zeros(len(weights), dtype=np.float32)
    weight = _lp_based_solver(edge_mat, len(weights), len(edges), weights, cover)
    print(cover, weight)
    cover = np.where(cover >= 0.5)[0]
    weight = sum(weights[cover])
    return cover, weight


def task1() -> Tuple[List[List[int]], List[float]]:
    """
    Task 1. The best solution is always obtained by greedy set cover algorithm

    Returns
    -------
    Tuple[List[List[int]], List[float]]
        The graph data and the weights of the nodes.
    """
    graph = [
        [0, 1],
        [0, 2],
        [1, 2],
        [1, 3],
        [2, 3],
        [2, 4],
        [3, 4]
    ]
    weights = [1, 3, 3, 4, 2.1]
    return graph, weights


def task2() -> Tuple[List[List[int]], List[float]]:
    """
    See task1() for the description. except this time the best is from graph cover.

    Returns
    -------
    Tuple[List[List[int]], List[float]]
        The graph data and the weights of the nodes.
    """
    graph = [
        [0, 4],
        [1, 0],
        [1, 3],
        [1, 4],
        [2, 1],
        [2, 4]
    ]
    weights = [
        816,
        829,
        65,
        352,
        536
    ]
    return graph, weights


def task3() -> Tuple[List[List[int]], List[float]]:
    """
    See task1() for the description. except this time the best is from Linear Programing based.

    Returns
    -------
    Tuple[List[List[int]], List[float]]
        The graph data and the weights of the nodes.
    """
    graph = [
        [0, 1],
        [0, 4],
        [1, 2],
        [1, 3],
        [2, 0],
        [4, 3]
    ]
    weights = [
        240,
        127,
        28,
        189,
        645
    ]
    return graph, weights


def main_random():
    len_data = 8
    max_node = 5
    while True:
        data = np.random.randint(0, max_node, (len_data, 2))
        # remove duplicates
        data = np.unique(data, axis=0)
        # remove self loops
        data = data[data[:, 0] != data[:, 1]]
        # remove not exist nodes
        for i in range(max_node):
            if i not in data.flatten():
                data -= (data > i).astype(int)

        weights = np.random.random(len(np.unique(data.flatten()))) * 10
        graph = data
        g_set = graph_set_convert(graph)

        def approx_eq(a, b):
            return abs(a - b) < 1e-5

        def same_cover(a, b):
            return set(a) == set(b)

        cover1, weight1 = lp_based_solve(graph, weights)
        cover2, weight2 = set_cover_greedy(g_set, weights, lambda *_: 1)
        if not approx_eq(weight1, weight2) or not same_cover(cover1, cover2):
            if weight1 > weight2:
                continue
        else:
            continue
        (cover3, weight3), _ = iterate_graph(graph, weights)
        if not approx_eq(weight1, weight3) or not same_cover(cover1, cover3):
            if weight1 > weight3:
                continue
        else:
            continue
        print(data)
        print(weights)
        print(cover1, weight1)
        print(cover2, weight2)
        print(cover3, weight3)
        draw_graph(g_set, weights, 'green')
        return


def main():
    graph, weights = task3()
    g_set = graph_set_convert(graph)
    print('The graph is:')
    draw_graph(g_set, weights, 'green')
    print('The greedy set solution is:')
    cover, weight = set_cover_greedy(g_set, weights, lambda *_: 1)
    print(cover, weight)
    print('The lp based solution is:')
    # cover, weight = ip_solve(graph, weights)
    cover, weight = lp_based_solve(graph, weights)
    print(cover, weight)
    print('The greedy graph solution is:')
    (cover, weight), _ = iterate_graph(graph, weights)
    print(cover, weight)


if __name__ == '__main__':
    main()
