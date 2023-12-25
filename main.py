from dataclasses import dataclass
import numpy as np
from sympy import symbols, lambdify, sympify
import matplotlib.pyplot as plt


@dataclass
class Point:
    x: float
    y: float


def create_exc_func(func: str) -> callable:
    x = symbols('x')
    parsed_expr = sympify(func)
    return lambdify(x, parsed_expr, 'numpy')


def get_intersection(A: Point, B: Point, l: float) -> Point:
    m = ((A.y - B.y) - l * (A.x - B.x)) / (2 * l)
    return Point(m + A.x, A.y - m * l)


def shubert_piyavskii(func: callable, a: float, b: float, l: float, epsilon=0.001):
    plt.plot(dots, list(map(func, dots)))
    plt.grid()

    A = Point(a, func(a))
    B = Point(b, func(b))

    g_0 = lambda x: func(a) - l * abs(x - a)
    g_1 = lambda x: func(b) - l * abs(x - b)
    p_1 = lambda x: max(g_0(x), g_1(x))
    plt.plot(dots, list(map(p_1, dots)))

    u = get_intersection(A, B, l)
    left_u, right_u = get_intersection(A, u, l), get_intersection(u, B, l)
    points = [A, left_u, u, right_u, B]

    plt.scatter([point.x for point in points], [point.y for point in points])

    delta = 10 ** 10_000
    while delta > epsilon:
        min_point_idx = np.argmin([point.y for point in points])

        min_point = Point(points[min_point_idx].x, func(points[min_point_idx].x))
        delta = min_point.y - points[min_point_idx].y

        left_point = get_intersection(points[min_point_idx - 1], min_point, l)
        right_point = get_intersection(min_point, points[min_point_idx + 1], l)

        points.pop(min_point_idx)

        points.insert(min_point_idx, left_point)

        points.insert(min_point_idx, min_point)

        points.insert(min_point_idx, right_point)

    plt.show()


a, b = -1.5, 1.5
dots = np.linspace(a, b, 1000)

t = '(1.5 - x + x * 0.5)^2 + (2.25 - x + x*0.5^2)^2 + (2.625 - x + x * 0.5^3)^2'
func = '(x^2-1)^2+x'
func = create_exc_func(func)
# print(piyavskii_method(func, a, b, 5, 0.001))
shubert_piyavskii(func, a, b, 5)
