from typing import Literal

import numpy as np

# Linear Programming Packages
from scipy.optimize import linprog
import pulp

import matplotlib.pyplot as plt

from scipy.io import loadmat


def task1_1() -> None:
    """
    A Simple example of LP with a single optimal solution

    \min(x+y)
    s.t. 2x+y >= 4
         x+2y >= 4
         x,y >= 0
    """
    # SciPy
    # Objective function
    c = np.array([1, 1])
    # Constraints
    A_ub = np.array([[-2, -1], [-1, -2]])
    b_ub = np.array([-4, -4])
    # Bounds
    bounds = [(0, None), (0, None)]
    # Solve
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds)
    print('SciPy:')
    print(res)

    print('\n\n----------------------\n\n')
    # PuLP
    # Create Variables
    x_1 = pulp.LpVariable('x_1', lowBound=0)
    x_2 = pulp.LpVariable('x_2', lowBound=0)
    # Create Problem
    prob = pulp.LpProblem('Task1_1', pulp.LpMinimize)
    # Objective Function
    prob += x_1 + x_2
    # Constraints
    prob += 2 * x_1 + x_2 >= 4
    prob += x_1 + 2 * x_2 >= 4
    # Solve
    status = prob.solve()
    print('PuLP:')
    print(pulp.LpStatus[status])
    print('x_1 =', pulp.value(x_1))
    print('x_2 =', pulp.value(x_2))

    # Plot problem
    x = np.linspace(0, 4, 100)
    y1 = (4 - 2 * x) / 1
    y2 = (4 - x) / 2
    plt.plot(x, y1, label='2x+y>=4')
    plt.plot(x, y2, label='x+2y>=4')
    plot_upper_bound = 4
    fill_sec_lower = np.maximum(y1, y2)
    fill_sec_upper = np.ones_like(fill_sec_lower) * plot_upper_bound
    plt.fill_between(x, fill_sec_lower, fill_sec_upper, alpha=0.2)
    plt.scatter(res.x[0], res.x[1], label='Optimal Solution HiGHS')
    plt.scatter(pulp.value(x_1), pulp.value(x_2), label='Optimal Solution CBC')
    plt.xlim(0, plot_upper_bound)
    plt.ylim(0, plot_upper_bound)
    plt.legend()
    plt.show()


def task1_2() -> None:
    """
    A simple example of LP with an optimal set on a line

    \min(x+y)
    s.t. 2x+y >= 4
         x+2y >= 4
         x+y >= 3
         x,y >= 0

    """
    # SciPy
    # Objective function
    c = np.array([1, 1])
    # Constraints
    A_ub = np.array([[-2, -1], [-1, -2], [-1, -1]])
    b_ub = np.array([-4, -4, -3])
    # Bounds
    bounds = [(0, None), (0, None)]
    # Solve
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds)
    print('SciPy:')
    print(res)

    print('\n\n----------------------\n\n')
    # PuLP
    # Create Variables
    x_1 = pulp.LpVariable('x_1', lowBound=0)
    x_2 = pulp.LpVariable('x_2', lowBound=0)
    # Create Problem
    prob = pulp.LpProblem('Task1_1', pulp.LpMinimize)
    # Objective Function
    prob += x_1 + x_2
    # Constraints
    prob += 2 * x_1 + x_2 >= 4
    prob += x_1 + 2 * x_2 >= 4
    prob += x_1 + x_2 >= 3
    # Solve
    status = prob.solve()
    print('PuLP:')
    print(pulp.LpStatus[status])
    print('x_1 =', pulp.value(x_1))
    print('x_2 =', pulp.value(x_2))

    # Plot problem
    x = np.linspace(0, 4, 100)
    y1 = (4 - 2 * x) / 1
    y2 = (4 - x) / 2
    y3 = (3 - x) / 1
    plt.plot(x, y1, label='2x+y>=4')
    plt.plot(x, y2, label='x+2y>=4')
    plt.plot(x, y3, label='x+y>=3')
    plot_upper_bound = 4
    fill_sec_lower = np.maximum(y1, y2)
    fill_sec_lower = np.maximum(fill_sec_lower, y3)
    fill_sec_lower = np.maximum(fill_sec_lower, np.zeros_like(fill_sec_lower))
    fill_sec_upper = np.ones_like(fill_sec_lower) * plot_upper_bound
    plt.fill_between(x, fill_sec_lower, fill_sec_upper, alpha=0.2)
    plt.scatter(res.x[0], res.x[1], label='Optimal Solution HiGHS')
    plt.scatter(pulp.value(x_1), pulp.value(x_2), label='Optimal Solution CBC')
    plt.xlim(0, plot_upper_bound)
    plt.ylim(0, plot_upper_bound)
    plt.legend()
    plt.show()


def task1_3() -> None:
    """
    A simple example of LP with an optimal set on a plane

    \min(x+y+z)
    s.t. 2x+y+z >= 4
         x+2y+z >= 4
         x+y+2z >= 4
         x+y+z >= 3
         x,y,z >= 0

    """
    class AxWrapper3D:
        def __init__(self, ax):
            self.ax = ax

        def __getattr__(self, item):
            attr = getattr(self.ax, item)
            if callable(attr):
                def wrapper(*args, **kwargs):
                    ret = attr(*args, **kwargs)
                    if hasattr(ret, '_facecolor3d'):
                        ret._facecolors2d = ret._facecolor3d
                        ret._edgecolors2d = ret._edgecolor3d
                    return ret
                return wrapper
            return attr

    # SciPy
    # Objective function
    c = np.array([1, 1, 1])
    # Constraints
    A_ub = np.array([[-2, -1, -1], [-1, -2, -1], [-1, -1, -2], [-1, -1, -1]])
    b_ub = np.array([-3.5, -3.5, -3.5, -3])
    # Bounds
    bounds = [(0, None), (0, None), (0, None)]
    # Solve
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds)
    print('SciPy:')
    print(res)

    print('\n\n----------------------\n\n')
    # PuLP
    # Create Variables
    x_1 = pulp.LpVariable('x_1', lowBound=0)
    x_2 = pulp.LpVariable('x_2', lowBound=0)
    x_3 = pulp.LpVariable('x_3', lowBound=0)
    # Create Problem
    prob = pulp.LpProblem('Task1_1', pulp.LpMinimize)
    # Objective Function
    prob += x_1 + x_2 + x_3
    # Constraints
    prob += 2 * x_1 + x_2 + x_3 >= 3.5
    prob += x_1 + 2 * x_2 + x_3 >= 3.5
    prob += x_1 + x_2 + 2 * x_3 >= 3.5
    prob += x_1 + x_2 + x_3 >= 3
    # Solve
    status = prob.solve()
    print('PuLP:')
    print(pulp.LpStatus[status])
    print('x_1 =', pulp.value(x_1))
    print('x_2 =', pulp.value(x_2))
    print('x_3 =', pulp.value(x_3))

    # Plot problem
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax = AxWrapper3D(ax)
    x = np.linspace(0, 4, 100)
    y = np.linspace(0, 4, 100)
    X, Y = np.meshgrid(x, y)
    Z1 = (3.5 - 2 * X - Y) / 1
    Z2 = (3.5 - X - 2 * Y) / 1
    Z3 = (3.5 - X - Y) / 2
    Z4 = (3 - X - Y) / 1
    ax.plot_surface(X, Y, Z1, alpha=0.2, label='2x+y+z>=3.5')
    ax.plot_surface(X, Y, Z2, alpha=0.2, label='x+2y+z>=3.5')
    ax.plot_surface(X, Y, Z3, alpha=0.2, label='x+y+2z>=3.5')
    ax.plot_surface(X, Y, Z4, alpha=0.2, label='x+y+z>=3')
    plot_upper_bound = 4
    ax.scatter(res.x[0], res.x[1], res.x[2], label='Optimal Solution HiGHS')
    ax.scatter(pulp.value(x_1), pulp.value(x_2), pulp.value(x_3), label='Optimal Solution CBC')
    ax.set_xlim(0, plot_upper_bound)
    ax.set_ylim(0, plot_upper_bound)
    ax.set_zlim(0, plot_upper_bound)
    ax.legend()
    plt.show()


def task2(mat_file: str, solver: Literal['scipy', 'pulp']) -> None:
    """
    Run the problem defined in the .mat file, should contain three matrix:
        A: coefficient matrix, maximize, (1, n)
        B: constraint matrix, minimize, (m, n)
        C: rhs, (m, 1)
    Parameters
    ----------
    mat_file : str
        Path to the .mat
    solver : Literal['scipy', 'pulp']
        Solver to use
    """
    # Load data
    data = loadmat(mat_file)
    A = data['A']
    B = data['B']
    C = data['C']
    # Solve
    if solver == 'scipy':
        bounds = [(0, None) for _ in range(A.shape[1])]
        print('Init SciPy')
        res = linprog(-A, B, C, bounds=bounds)
        print('SciPy:')
        print('f(x) =', -res.fun)
    elif solver == 'pulp':
        # Create Variables
        x = [pulp.LpVariable(f'x_{i}', lowBound=0) for i in range(A.shape[1])]
        # Create Problem
        prob = pulp.LpProblem('Task2', pulp.LpMaximize)
        # Objective Function
        prob += pulp.lpDot(A, x)
        # Constraints
        for i in range(B.shape[0]):
            prob += pulp.lpDot(B[i], x) <= C[i]
        print('Init PuLP')
        # Solve
        status = prob.solve()
        print('PuLP:')
        print(pulp.LpStatus[status])
        print('f(x) =', sum([pulp.value(x_i) for x_i in x]))
    else:
        raise ValueError(f'Unknown solver {solver}')


if __name__ == '__main__':
    task1_3()

