import random
import time

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sympy import symbols, lambdify, sympify


def create_exc_func(func: str) -> callable:
    x = symbols('x')
    y = symbols('y')
    parsed_expr = sympify(func)
    return lambdify([x, y], parsed_expr, 'numpy')


class Species:

    def __init__(self, x_boarder: tuple, y_boarder: tuple, random_creature=True):
        self._x = random.uniform(*x_boarder)
        self._y = random.uniform(*y_boarder)

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y


class ManuelSpecies(Species):

    def __init__(self, x: float, y: float):
        self._x = x
        self._y = y


class GeneticAlgo:

    def __init__(self, func: callable, x_boarder: tuple, y_boarder: tuple, epsilon=0.00001, n_species=10):
        self._func = func
        self._x_boarder = x_boarder
        self._y_boarder = y_boarder
        self._eps = epsilon
        self._population = [Species(x_boarder, y_boarder) for _ in range(n_species)]
        self._n_species = n_species

    def __sort_species(self):
        self._population = sorted(self._population, key=lambda specie: self._func(specie.x, specie.y), reverse=False)

    def __crossover(self):
        # 0.4 всех особей оставляем оба гена
        x_genes = [specie.x for specie in self._population[:int(len(self._population) * 0.4)]]
        y_genes = [specie.y for specie in self._population[:int(len(self._population) * 0.4)]]

        # 0.3 оставляем только один случайный ген
        x_random_genes = [specie.x for specie in self._population[int(len(self._population) * 0.4):]]
        y_random_genes = [specie.y for specie in self._population[int(len(self._population) * 0.4):]]
        random.shuffle(x_random_genes)
        random.shuffle(y_random_genes)

        x_random_genes = x_random_genes[:len(x_random_genes) // 2]
        y_random_genes = y_random_genes[:len(y_random_genes) // 2]

        x_genes.extend(x_random_genes)
        y_genes.extend(y_random_genes)

        random.shuffle(x_genes)
        random.shuffle(y_genes)

        new_species = [ManuelSpecies(x, y) for x, y in zip(x_genes, y_genes)]
        return new_species

    def is_quality_improved(self, results, k, epsilon) -> bool:
        if len(results) < k + 1:
            return True

        k_results = results[-k:-1]

        if max([abs(results[-1] - k_result) for k_result in k_results]) < epsilon:
            return False
        return True

    def start(self):
        epoch = 0

        history = []
        hist_results = []

        # Ранжируем особей
        self.__sort_species()

        while epoch < 1000 and self.is_quality_improved(hist_results, 10, self._eps):
            # Удаляем 0.3 особей
            self._population = self._population[:int(len(self._population) * 0.7)]

            # Из оставшихся производим новые особи
            new_species = self.__crossover()

            # Не тронутыми остаются 0.3
            self._population = self._population[:int(len(self._population) * 0.3)]

            self._population.extend(new_species)

            self._population.extend([Species(self._x_boarder, self._y_boarder)
                                     for _ in range(self._n_species - len(self._population))])

            # Ранжируем новую популяцию
            self.__sort_species()

            epoch += 1

            history.append(self._population[0])
            hist_results.append(self._func(self._population[0].x, self._population[1].y))

        print(epoch)

        print(history[-1].x, history[-1].y)
        print(hist_results[-1])

        #self.visualize(history)

    def visualize(self, points: list):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Создаем сетку для функции
        x = np.linspace(-5, 5, 100)
        y = np.linspace(-5, 5, 100)
        x, y = np.meshgrid(x, y)
        z = self._func(x, y)

        # Визуализируем функцию
        ax.plot_surface(x, y, z, alpha=0.5, rstride=100, cstride=100)

        # Визуализируем точки движения алгоритма
        xs, ys = zip(*[(point.x, point.y) for point in points])
        zs = [self._func(point.x, point.y) for point in points]
        ax.scatter(xs, ys, zs, '*', color='red')

        plt.show()


if __name__ == '__main__':
    func = '(1.5 - x + x * 0.5)^2 + (2.25 - x + x*0.5^2)^2 + (2.625 - x + x * 0.5^3)^2'
    func = create_exc_func(func)
    instance = GeneticAlgo(func, (-4.5, 4.5), (-4.5, 4.5))

    start_time = time.time()
    instance.start()
    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time:.5f} seconds")