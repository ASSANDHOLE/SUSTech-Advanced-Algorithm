from typing import List, Tuple, Callable, Any

import numpy as np

from algorithms import set_cover_int, set_cover_float


def set_cover_greedy(subsets: List[List[int]],
                     weights: List[int | float] | np.ndarray,
                     tie_breaking_fn: Callable[
                         [list, list | np.ndarray, list | np.ndarray, int],
                         Any
                     ]) -> \
        Tuple[List[int], int | float]:
    """
    Parameters
    ----------
    subsets : List[List[int]]
        The subsets to cover.
    weights : List[int | float] | np.ndarray
        The weights of the subsets.
    tie_breaking_fn : Callable[[list, list | np.ndarray, list | np.ndarray int], Any]
        A function that takes the subsets, weights, covered, and the index of the subset,
          and returns a comparable as fitness.
        The tie breaking function. The smaller the value, the better the subset.
        If the value is the same, the subset with the smaller index is chosen.

    Returns
    -------
    Tuple[List[int], int | float]
        The indices of the subsets to cover and the total weight.
    """
    all_elements = set()
    for subset in subsets:
        all_elements.update(subset)
    covered_elements = [False for _ in range(len(all_elements))]
    covered_subsets = []
    total_weight = 0
    while not all(covered_elements):
        best_subset = None
        best_subsets = []
        best_weight = float('inf')
        for i, subset in enumerate(subsets):
            if i in covered_subsets:
                continue
            weight = weights[i]
            new_covered_element_len = len([e for e in subset if not covered_elements[e]])
            if new_covered_element_len == 0:
                continue
            weight /= new_covered_element_len
            if weight < best_weight:
                best_subset = i
                best_subsets = [i]
                best_weight = weight
            elif weight == best_weight:
                best_subsets.append(i)
        if len(best_subsets) > 1:
            tie_breaking_arr = [tie_breaking_fn(subsets, weights, covered_elements, i) for i in best_subsets]
            tie_breaking_arr = list(zip(tie_breaking_arr, range(len(tie_breaking_arr))))
            tie_breaking_arr.sort()
            best_subset = best_subsets[tie_breaking_arr[0][1]]
        covered_subsets.append(best_subset)
        total_weight += weights[best_subset]
        for e in subsets[best_subset]:
            covered_elements[e] = True
    return covered_subsets, total_weight


def calc_hd(subsets: List[List[int]]) -> float:
    """
    Parameters
    ----------
    subsets : List[List[int]]
        The subsets.

    Returns
    -------
    float
        The Harmonic Function of the length of the largest subset.
    """
    max_len = max([len(s) for s in subsets])
    return sum([1 / i for i in range(1, max_len + 1)])


def task1_1() -> Tuple[List[List[int]], List[int]]:
    """
    Task 1.1 Simple example that a good solution is not acquired by the greedy algorithm

    Returns
    -------
    Tuple[List[List[int]], List[int]]
        The subsets and the weights.
    """
    subsets = [
        [0, 2],
        [1, 3],
        [0, 1]
    ]
    weights = [4, 4, 3.99]
    return subsets, weights


def task1_2() -> Tuple[List[List[int]], List[int]]:
    """
    Task 1.2 Interesting example that a good solution is not acquired by the greedy algorithm

    Returns
    -------
    Tuple[List[List[int]], List[int]]
        The subsets and the weights.
    """
    subsets = [
        [0, 1],
        [2],
        [0, 3],
        [1, 2]
    ]
    weights = [4, 4.01, 4.02, 4.02]
    return subsets, weights


def task2_1() -> Tuple[List[List[int]], List[int], List[int] | None, int | None]:
    """
    Task 2.1 Simple example that good and bad solutions are acquired by the greedy algorithm,
    with different tie breaking functions.

    Returns
    -------
    Tuple[List[List[int]], List[int], List[int] | None, int | None]
        The subsets and the weights. And Optional the optimal solution.
    """
    subsets = [
        [0, 2],
        [0, 1],
        [2, 3],
        [4, 5],
        [6, 7],
        [4, 6],
    ]
    weights = [1] * len(subsets)
    opt = list(range(2, len(subsets)))
    return subsets, weights, opt, len(opt)


def task2_2() -> Tuple[List[List[int]], List[int], List[int] | None, int | None]:
    """
    Task 2.2 Interesting example that good and bad solutions are acquired by the greedy algorithm,
    with different tie breaking functions.

    Returns
    -------
    Tuple[List[List[int]], List[int], List[int] | None, int | None]
        The subsets and the weights. And Optional the optimal solution.
    """
    subsets = [
        [0, 3, 6],
        [0, 4, 7],
        [9, 12, 15],
        [9, 13, 16],
        [1, 10],
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
        [9, 10, 11],
        [12, 13, 14],
        [15, 16, 17],
    ]
    weights = [1] * len(subsets)
    opt = list(range(5, len(subsets)))
    return subsets, weights, opt, len(opt)


def more_uncovered_tie_breaking(
        subsets: List[List[int]],
        weights: List[int | float] | np.ndarray,
        covered_elements: List[int] | np.ndarray,
        idx: int
) -> Tuple[float, float]:
    """
    The more uncovered elements in the subset, the better the subset.
    @See set_cover_greedy
    """
    return -len([e for e in subsets[idx] if not covered_elements[e]]), idx


def less_uncovered_tie_breaking(
        subsets: List[List[int]],
        weights: List[int | float] | np.ndarray,
        covered_elements: List[int] | np.ndarray,
        idx: int
) -> Tuple[float, float]:
    """
    The less uncovered elements in the subset, the better the subset.
    @See set_cover_greedy
    """
    l, i = more_uncovered_tie_breaking(subsets, weights, covered_elements, idx)
    return -l, i


def more_elements_tie_breaking(
        subsets: List[List[int]],
        weights: List[int | float] | np.ndarray,
        covered_elements: List[int] | np.ndarray,
        idx: int
) -> Tuple[float, float]:
    """
    The more elements in the subset, the better the subset.
    @See set_cover_greedy
    """
    return -len(subsets[idx]), idx


def less_elements_tie_breaking(
        subsets: List[List[int]],
        weights: List[int | float] | np.ndarray,
        covered_elements: List[int] | np.ndarray,
        idx: int
) -> Tuple[float, float]:
    """
    The fewer elements in the subset, the better the subset.
    @See set_cover_greedy
    """
    l, i = more_elements_tie_breaking(subsets, weights, covered_elements, idx)
    return -l, i


def more_weight_tie_breaking(
        subsets: List[List[int]],
        weights: List[int | float] | np.ndarray,
        covered_elements: List[int] | np.ndarray,
        idx: int
) -> Tuple[float, float]:
    """
    The more weight in the subset, the better the subset.
    @See set_cover_greedy
    """
    return -weights[idx], idx


def less_weight_tie_breaking(
        subsets: List[List[int]],
        weights: List[int | float] | np.ndarray,
        covered_elements: List[int] | np.ndarray,
        idx: int
) -> Tuple[float, float]:
    """
    The less weight in the subset, the better the subset.
    @See set_cover_greedy
    """
    l, i = more_weight_tie_breaking(subsets, weights, covered_elements, idx)
    return -l, i


def more_unique_tie_breaking(
        subsets: List[List[int]],
        weights: List[int | float] | np.ndarray,
        covered_elements: List[int] | np.ndarray,
        idx: int
) -> Tuple[float, float]:
    """
    The more rare elements in the subset, the better the subset.
    @See set_cover_greedy
    """
    all_elements = set()
    for subset in subsets:
        all_elements.update(subset)
    elem_count = [0 for _ in range(len(all_elements))]
    for subset in subsets:
        for e in subset:
            elem_count[e] += 1
    return -np.mean([1 / elem_count[e]**2 for e in subsets[idx]]), idx


def less_unique_tie_breaking(
        subsets: List[List[int]],
        weights: List[int | float] | np.ndarray,
        covered_elements: List[int] | np.ndarray,
        idx: int
) -> Tuple[float, float]:
    """
    The less rare elements in the subset, the better the subset.
    @See set_cover_greedy
    """
    l, i = more_unique_tie_breaking(subsets, weights, covered_elements, idx)
    return -l, i


def main_task1():
    subsets, weights = task1_2()
    optimal_subsets, optimal_weight = set_cover_float(subsets, weights)
    greedy_subsets, greedy_weight = set_cover_greedy(subsets, weights, lambda s, w, c, i: i)
    hd = calc_hd(subsets)
    print(f'Optimal solution: {optimal_subsets}, {optimal_weight}')
    print(f'Greedy solution: {greedy_subsets}, {greedy_weight}')
    print(f'Harmonic Function of the length of the largest subset: {hd}')
    print(f'Optimal solution is {greedy_weight / optimal_weight:.2f} times better than the greedy solution')


def main_task2():
    subsets, weights, *opt = task2_2()
    tie_breaking_fns = [
        more_uncovered_tie_breaking,
        less_uncovered_tie_breaking,
        more_elements_tie_breaking,
        less_elements_tie_breaking,
        more_weight_tie_breaking,
        less_weight_tie_breaking,
        more_unique_tie_breaking,
        less_unique_tie_breaking,
    ]
    if opt is None:
        best_subsets, best_weight = set_cover_int(subsets, weights)
    else:
        best_subsets, best_weight = opt
    hd = calc_hd(subsets)
    print(f'Best solution: {best_subsets}, {best_weight}')
    print(f'Harmonic Function of the length of the largest subset: {hd}\n')
    for tie_breaking_fn in tie_breaking_fns:
        greedy_subsets, greedy_weight = set_cover_greedy(subsets, weights, tie_breaking_fn)
        print(f'Tie breaking function: {tie_breaking_fn.__name__}')
        print(f'Greedy solution: {greedy_subsets}, {greedy_weight}')
        print(f'{greedy_weight / best_weight:.2f} times better\n')


if __name__ == '__main__':
    main_task1()
