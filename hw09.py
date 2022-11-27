from typing import List, Tuple
from itertools import permutations

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from algorithms import set_cover_int

from hw08 import set_cover_greedy

G = 'green'
Y = 'yellow'


class GraphWrapper(nx.Graph):

    def __init__(self):
        super().__init__()

    def add_edge(self, u_of_edge, v_of_edge, **attr):
        if 'weight' in attr and 'length' not in attr:
            attr['length'] = attr['weight']
        super().add_edge(u_of_edge, v_of_edge, **attr)


def graph_set_convert(data: List[List[int]] | np.ndarray) -> List[List[int]]:
    """
    Convert the graph to a set.

    Parameters
    ----------
    data : List[List[int]] | np.ndarray
        The data. [[item1, item2, ? if node], ...]
        Note that if the input is a graph, the array in data must be length 2.

    Returns
    -------
    List[List[int]]
        The set data. [[item1, item2, ? if edge], ...]
    """
    arr = []
    for i in range(len(data)):
        arr.extend(data[i])
    items = [[] for _ in range(max(arr) + 1)]
    for i in range(len(data)):
        for j in range(len(data[i])):
            items[data[i][j]].append(i)
    return items


def draw(data: List[List[int]] | np.ndarray,
         weights: List[int | float] | np.ndarray,
         colors: List[str] | str,
         edge_color: str = 'red',
         show: bool = True) -> None:
    """
    Draw the graph.

    Parameters
    ----------
    data : List[List[int]] | np.ndarray
        The graph data. [[edge1, edge2, ...], ...]
    weights : List[int | float] | np.ndarray
        The weights of the nodes. [weight1, weight2, ...]
    colors : List[str] | str
        The colors of the nodes. [color1, color2, ...] or color
    edge_color : str
        The color of the edges, by default 'red'
    show : bool
        Whether to show the graph, by default True

    """
    items = graph_set_convert(data)

    if type(weights[0]) not in (int, np.int32, np.int64):
        nodes = [f'{chr(i + 65)}:{weights[i]:.2f}' for i in range(len(data))]
    else:
        nodes = [f'{chr(i + 65)}:{weights[i]}' for i in range(len(data))]

    graph = GraphWrapper()
    for edge in items:
        graph.add_edge(nodes[edge[0]], nodes[edge[1]], weight=1)

    pos = nx.spring_layout(graph)

    options = {
        'node_color': colors,
        'node_size': 5000,
        'width': 3,
        'with_labels': True,
        'edge_color': edge_color,
        'font_size': 20,
    }

    nx.draw(graph, pos, **options)
    edge_labels = nx.get_edge_attributes(graph, 'label')
    edge_options = {
        'font_color': 'black',
        'font_size': 20,
    }
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, **edge_options)
    plt.show() if show else None


def graph_cover_greedy(data: List[List[int]] | np.ndarray,
                       weights: List[int | float] | np.ndarray) -> \
        Tuple[List[int], int | float]:
    """
    Find a set cover of the graph.

    Parameters
    ----------
    data : List[List[int]] | np.ndarray
        The graph data. [[node1, node2], ...]
    weights : List[int | float] | np.ndarray
        The weights of the nodes. [weight1, weight2, ...]

    Returns
    -------
    Tuple[List[int], int | float]
        The index of the graph cover and the weight.
    """
    eps = 1e-6

    def approx_equal(a, b):
        return abs(a - b) < eps

    nodes = np.array(weights, dtype=type(weights[0]))
    node_done = np.zeros_like(nodes, dtype=bool)
    edge_done = np.zeros(len(data), dtype=bool)
    while not all(edge_done):
        for i, ed in enumerate(edge_done):
            if ed:
                continue
            edge_done[i] = True
            if node_done[data[i][0]] or node_done[data[i][1]]:
                continue
            if approx_equal(nodes[data[i][0]], nodes[data[i][1]]):
                node_done[data[i][0]] = True
                node_done[data[i][1]] = True
                continue
            if nodes[data[i][0]] > nodes[data[i][1]]:
                node_done[data[i][0]] = True
                nodes[data[i][0]] -= nodes[data[i][1]]
            else:
                node_done[data[i][1]] = True
                nodes[data[i][1]] -= nodes[data[i][0]]
    return np.where(node_done)[0].tolist(), np.array(weights)[node_done].sum()


def iterate_graph(data: List[List[int]] | np.ndarray,
                  weights: List[int | float] | np.ndarray) -> \
        Tuple[Tuple[List[int], int | float], Tuple[List[int], int | float]]:
    """
    See `graph_cover_greedy`

    Returns
    -------
    Tuple[Tuple[List[int], int | float], Tuple[List[int], int | float]]
        The best and the worst:
            The index of the graph cover and the weight.
    """
    best = None, np.inf
    worst = None, -np.inf
    for perm in permutations(data):
        cover, weight = graph_cover_greedy(perm, weights)
        if weight < best[1]:
            best = cover, weight
        if weight > worst[1]:
            worst = cover, weight
    return best, worst


def main_random():
    len_data = 8
    max_node = 5
    data = np.random.randint(0, max_node, (len_data, 2))
    # remove duplicates
    data = np.unique(data, axis=0)
    # remove self loops
    data = data[data[:, 0] != data[:, 1]]
    # remove not exist nodes
    for i in range(max_node):
        if i not in data.flatten():
            data -= (data > i).astype(int)

    print('data:\n', data)

    weights = np.random.random(len(np.unique(data.flatten()))) * 10
    (gra_idx, gra_w), _ = iterate_graph(data, weights)
    data = graph_set_convert(data)
    set_idx, set_w = set_cover_greedy(data, weights, lambda *_: 1)
    best_idx, best_w = set_cover_int(data, weights)
    print(f'gra_idx:  {gra_idx}, gra_w:  {gra_w}')
    print(f'set_idx:  {set_idx}, set_w:  {set_w}')
    print(f'best_idx: {best_idx}, best_w: {best_w}')
    colors = Y
    draw(data, weights, colors)
    plt.show()


def main_deterministic():
    data = [
        [0, 1],
        [0, 3],
        [1, 3],
        [0, 2],
        [1, 4]
    ]
    weights = [
        186,
        283,
        438,
        48,
        278
    ]
    (gra_idx, gra_w), _ = iterate_graph(data, weights)
    data = graph_set_convert(data)
    set_idx, set_w = set_cover_greedy(data, weights, lambda *_: 1)
    best_idx, best_w = set_cover_int(data, weights)
    print(f'gra_idx:  {gra_idx}, gra_w:  {gra_w}')
    print(f'set_idx:  {set_idx}, set_w:  {set_w}')
    print(f'best_idx: {best_idx}, best_w: {best_w}')
    colors = [Y, G, G, Y, Y]
    draw(data, weights, colors)
    plt.show()


if __name__ == '__main__':
    main_deterministic()
