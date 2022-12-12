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
    ]
    route_pairs = [
        (4, 1),
        (5, 2),
        (6, 3),
        (0, 3)
    ]
    return edges, route_pairs, 1


def main():
    # great way to draw graphs: https://csacademy.com/app/graph_editor/
    edges, route_pairs, cap = task1_3()
    data = disjoint_path_problem(edges, route_pairs, cap)
    best_order, best_path = data[0]
    print(f'best_order: {best_order}, best_path: {best_path}')
    worst_order, worst_path = data[1]
    print(f'worst_order: {worst_order}, worst_path: {worst_path}')
    # nodes = graph_set_convert(edges)
    # draw_graph(nodes, [1] * len(nodes), 'blue')


if __name__ == '__main__':
    main()
