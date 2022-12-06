import ctypes
from typing import List, Tuple, Literal

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from hw11 import _hw12_lp_based_solver


def lp_based_solve(machines: List[List[int]],
                   jobs_weights: List[float],
                   solver_name: Literal['GLOP', 'SCIP', 'CBC'] = 'GLOP'
                   ) -> Tuple[float, np.ndarray]:
    """
    The Linear Programming based solver for the generalized load balancing problem.

    Parameters
    ----------
    machines : List[List[int]]
        The doable jobs for each machine. [[job_id1, job_id2, ...], ...]
        The Length of the inner list can be different.
    jobs_weights : List[float] | np.ndarray
        The weights of the jobs. [weight1, weight2, ...]
    solver_name : Literal['GLOP', 'SCIP', 'CBC']
        The solver name.

    Returns
    -------
    Tuple[float, np.ndarray]
        The total of the graph cover and the x_ij (2d array) for i in machines j in jobs.
    """
    jobs_weights = np.array(jobs_weights, dtype=np.float32)
    jobs_num = len(jobs_weights)
    machine_shape = [len(m) for m in machines]
    machine_num = len(machine_shape)
    machines_input_array = np.zeros((sum(machine_shape),), dtype=np.int32)
    cur = 0
    for m in machines:
        machines_input_array[cur:cur + len(m)] = m
        cur += len(m)
    machine_shape = np.array(machine_shape, dtype=np.int32)
    ret = np.zeros((machine_num * jobs_num,), dtype=np.float32)
    solver_name = ctypes.c_char_p(solver_name.encode('utf-8'))
    weight = _hw12_lp_based_solver(jobs_weights, jobs_num, machines_input_array,
                                   machine_shape, machine_num, ret, solver_name)
    ret = ret.reshape((machine_num, jobs_num))
    return weight, ret


def cycle_in_graph_detect(x: np.ndarray) -> bool:
    """
    Detect if there is a cycle in the graph.

    Parameters
    ----------
    x : np.ndarray
        The graph data. Machine - Job. [[m1_j1, m1_j2, ...], [m2_j1, m2_j2, ...], ...]
        If m_j > 0, there is an edge from m to j.

    Returns
    -------
    bool
        If there is a cycle in the graph.
    """
    graph = nx.DiGraph()
    for i, m in enumerate(x):
        for j, v in enumerate(m):
            if v > 0:
                graph.add_edge(str(i), j)
    try:
        ret = nx.find_cycle(graph)
        print('the graph has a cycle')
        print(ret)
        return True
    except nx.NetworkXNoCycle:
        return False



def draw_graph(machines: List[List[int]],
               jobs_weights: List[float],
               solution: np.ndarray,
               text_only: bool = False) -> None:
    if text_only:
        # print machine
        for i, _ in enumerate(machines):
            print(f'M{i}')
        # print jobs
        for i, _ in enumerate(jobs_weights):
            print(f'J{i}')
        # print connections by solution
        for i, m in enumerate(solution):
            for j, v in enumerate(m):
                if v > 0:
                    print(f'M{i} J{j} {v:.2f}')
        return

    graph = nx.Graph()
    machine_ids = [f'M_{i}' for i in range(len(machines))]
    job_ids = [f'J_{i}' for i in range(len(jobs_weights))]
    graph.add_nodes_from(machine_ids, bipartite=0)
    graph.add_nodes_from(job_ids, bipartite=1)
    edges = []
    edge_labels = {}
    for i, m in enumerate(machines):
        for j in m:
            if solution[i, j] > 0:
                edge_labels[(machine_ids[i], job_ids[j])] = f'{solution[i, j]:.2f}'
                edges.append((machine_ids[i], job_ids[j]))
    graph.add_edges_from(edges)
    pos = {}
    for i, m in enumerate(machine_ids):
        pos[m] = (0, i + 0.5)
    max_y = len(machine_ids) + 1
    ratio = max_y / (len(job_ids) + 1)
    for i, j in enumerate(job_ids):
        pos[j] = (1, i * ratio)
    nx.draw_networkx_nodes(graph, pos, nodelist=machine_ids, node_color='r', node_size=2000, alpha=0.8)
    nx.draw_networkx_nodes(graph, pos, nodelist=job_ids, node_color='b', node_size=1000, alpha=0.8)
    nx.draw_networkx_edges(graph, pos, edgelist=edges, width=1.0, alpha=0.5)
    labels = {}
    for i, m in enumerate(machines):
        labels[machine_ids[i]] = f'M{i}'
    for i, w in enumerate(jobs_weights):
        labels[job_ids[i]] = f'J{i}({w:.2f})'
    nx.draw_networkx_labels(graph, pos, labels, font_size=16)
    nx.draw_networkx_edge_labels(graph, pos,
                                 edge_labels=edge_labels,
                                 font_size=16)
    plt.axis('off')
    plt.show()


def task1() -> Tuple[List[List[int]], List[float]]:
    """
    Task 1: Load Balancing Problem (no cycle)

    Returns
    -------
    Tuple[List[List[int]], List[float]]
        The graph data and the weights.
    """
    jobs = [3, 6, 6, 4, 3, 6]
    machines = [
        [0, 1],
        [1, 2, 4],
        [2, 3, 4],
        [4, 5]
    ]
    return machines, jobs


def task2() -> Tuple[List[List[int]], List[float], np.ndarray]:
    """
    Task 2: Load Balancing Problem (with cycle)

    Returns
    -------
    Tuple[List[List[int]], List[float], np.ndarray]
        The graph data and the weights. and the solution.
    """
    jobs = [1, 3, 2]
    machines = [
        [0, 1, 2],
        [1, 2]
    ]
    solution = np.array([
        [1, 1, 1],
        [0, 2, 1]
    ], dtype=np.float32)
    return machines, jobs, solution


def task_random() -> Tuple[List[List[int]], List[float]]:
    max_jobs = 10
    max_machines = 8
    max_weight = 10
    machines = []
    for _ in range(np.random.randint(int(max_machines // 2), max_machines)):
        machines.append(np.random.choice(max_jobs, np.random.randint(2, max_jobs), replace=False).tolist())

    # eliminate non-connected jobs
    jobs = []
    for i in range(max_jobs):
        for m in machines:
            if i in m:
                jobs.append(i)
                break
    for m in machines:
        for i in range(len(m)):
            m[i] = jobs.index(m[i])

    jobs_weight = np.random.rand(len(jobs))
    jobs_weight -= np.min(jobs_weight)
    jobs_weight /= np.max(jobs_weight)
    jobs_weight *= (max_weight - 1)
    jobs_weight += 1
    jobs_weight = jobs_weight.tolist()
    return machines, jobs_weight


def main():
    mac, jobs, sol = task2()
    draw_graph(mac, jobs, sol, text_only=True)


if __name__ == '__main__':
    main()
