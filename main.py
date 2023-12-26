from dataclasses import dataclass
import numpy as np
from sympy import symbols, lambdify, sympify
import matplotlib.pyplot as plt

COLORS = ['r', 'g']


@dataclass
class Point:
    x: float
    y: float


def create_exc_func(func: str) -> callable:
    x = symbols('x')
    parsed_expr = sympify(func)
    return lambdify(x, parsed_expr, 'numpy')


def get_intersection(A: Point, B: Point, l: float) -> Point:
    # Точка пересечения
    x = (A.y - B.y) / (2 * l) + (A.x + B.x) / 2

    # Значение ломанной в данной точке
    y = B.y + l * (x - B.x)
    return Point(x, y)


def shubert_piyavskii(func: callable, a: float, b: float, l: float, epsilon=0.001, max_iter=10_000) -> float:
    A = Point(a, func(a))
    B = Point(b, func(b))

    g_0 = lambda x: func(a) - l * abs(x - a)
    g_1 = lambda x: func(b) - l * abs(x - b)
    p_1 = lambda x: max(g_0(x), g_1(x))

    u = get_intersection(A, B, l)

    line_points = [Point(A.x, p_1(A.x)), u, Point(B.x, p_1(B.x))]
    points = [A, Point(u.x, func(u.x)), B]

    delta = 10 ** 10_000
    n_iter = 0
    while delta > epsilon or n_iter == max_iter:
        # Индекс минимума ломаной
        min_line_point_idx = np.argmin([point.y for point in line_points])

        # Получение значений на ломаной и функции
        min_line_point = line_points[min_line_point_idx]
        min_point = points[min_line_point_idx]

        delta = min_point.y - min_line_point.y

        left_point = get_intersection(points[min_line_point_idx - 1], min_point, l)
        right_point = get_intersection(min_point, points[min_line_point_idx + 1], l)

        func_left_point = Point(left_point.x, func(left_point.x))
        func_right_point = Point(right_point.x, func(right_point.x))

        line_points[min_line_point_idx] = min_point
        line_points.insert(min_line_point_idx, left_point)
        line_points.insert(min_line_point_idx + 2, right_point)

        points.insert(min_line_point_idx, func_left_point)
        points.insert(min_line_point_idx + 2, func_right_point)

        n_iter += 1

        plt.plot(
            [left_point.x, min_point.x, right_point.x],
            [left_point.y, min_point.y, right_point.y],
            color=COLORS[n_iter % 2]
        )

    plt.plot(dots, list(map(func, dots)), label='target', color='blue')
    plt.plot(dots, list(map(p_1, dots)), color=COLORS[n_iter % 2])
    plt.grid()
    plt.show()

    return min_point


if __name__ == "__main__":
    # func = '(1.5 - x + x * 0.5)^2 + (2.25 - x + x*0.5^2)^2 + (2.625 - x + x * 0.5^3)^2'
    func = '(x^2-1)^2+x'
    a, b = -1.5, 1.5
    L = 5

    dots = np.linspace(a, b, 1000)
    func = create_exc_func(func)
    print(shubert_piyavskii(func, a, b, L))
