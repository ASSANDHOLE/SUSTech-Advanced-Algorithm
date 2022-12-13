from typing import Tuple, List

import numpy as np

from algorithms import disjoint_path_problem

from hw09 import graph_set_convert
from hw11 import draw_graph


def task1_1() -> Tuple[List[List[int]], List[Tuple[int, int]], int]:
    edges = [
        [0, 1],
        [0, 2],
        [0, 3],
        [4, 0],
        [5, 0],
        [6, 0],
    ]
    route_pairs = [
        (4, 1),
        (5, 2),
        (6, 3),
    ]
    return edges, route_pairs, 1


def task1_2() -> Tuple[List[List[int]], List[Tuple[int, int]], int]:
    pass


def task1_3() -> Tuple[List[List[int]], List[Tuple[int, int]], int]:
    edges = [
        [0, 1],
        [1, 2],
        [2, 3],
        [4, 0],
        [5, 1],
        [6, 2],
        [3, 7]
    ]
    route_pairs = [
        (4, 1),
        (5, 2),
        (6, 3),
        (0, 7)
    ]
    return edges, route_pairs, 1


def task2_1() -> Tuple[List[List[int]], List[Tuple[int, int]], int]:
    edges, route_pairs, cap = task1_3()
    return edges, route_pairs, 2


def task2_2() -> Tuple[List[List[int]], List[Tuple[int, int]], int]:
    n = 10
    edges = []
    for i in range(n - 1):
        edges.append([i, i + 1])
    route_pairs = [(0, n - 2), (1, n - 1)]
    for i in range(1, n - 3):
        new_node_idx = i + n - 1
        edges.append([new_node_idx, i])
        route_pairs.append((new_node_idx, i + 1))
    return edges, route_pairs, 2


def main():
    # great way to draw graphs: https://csacademy.com/app/graph_editor/
    edges, route_pairs, cap = task1_1()
    data = disjoint_path_problem(edges, route_pairs, cap)
    best_order, best_path = data[0]
    print(f'best_order: {best_order}, best_path: {best_path}')
    worst_order, worst_path = data[1]
    print(f'worst_order: {worst_order}, worst_path: {worst_path}')
    best_path_n = sum([1 for p in best_path if len(p) > 0])
    worst_path_n = sum([1 for p in worst_path if len(p) > 0])
    print(f'best_path_n: {best_path_n}, worst_path_n: {worst_path_n}')
    print(f'best_path_n / worst_path_n: {best_path_n / worst_path_n}')
    print(f'best_ratio: {(2*cap * len(edges)**(1/(cap+1))) + 1}')


if __name__ == '__main__':
    main()
