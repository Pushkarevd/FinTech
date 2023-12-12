import re
from dataclasses import dataclass
from copy import deepcopy

import numpy as np


class TargetFunc:

    def __init__(self, func_string: str):
        func, target = func_string.split()
        self._func = func.strip()
        self._optimization_task = target.strip()

    @property
    def func(self):
        return self._func

    @property
    def target(self):
        return self._optimization_task


class Task:

    def __init__(self, target_task: TargetFunc, matrix_string: str):
        self._target_task = target_task
        self._A, self._b, self._x_matrix, self._x_target = self._prepare_matrix(matrix_string)

    def _prepare_matrix(self, matrix_string: str) -> tuple:
        """
        Функция для предобработки входных данных и формирования матрицы А и b
        :param matrix_string: входные данные, список ограничений
        :return: A, b матрицы
        """
        # Разделение строки на A и b
        prepared_strings = matrix_string.replace(' ', '') \
                               .split('\n')[1:-1]

        row_sign = [re.findall('=|<=|>=', row)[0] for row in prepared_strings]

        # Формирование b из строки уравнения
        b = np.array([float(re.split('=|<=|>=', x)[1])
                      for x in prepared_strings])

        # Нахождение максимального индекса x_i
        matrix_size = self._find_max_x(prepared_strings)

        # Инициализация матрицы
        matrix = [[0 for _ in range(matrix_size)]
                  for _ in range(len(b))]

        rows = [
            row if row[0] != 'x' else f'+{row}'
            for row in [(re.split('=|<=|>=', x)[0])
                        for x in prepared_strings]
        ]

        sign_map = {'-': -1, '+': 1}

        # Заполнение матрицы A
        for row_id, row in enumerate(rows):
            x_idx = [int(x) - 1 for x in re.findall(r'x_(\d+)', row)]
            coefs = re.split('x_[0-9]+', row)[:-1]

            coefs = [
                float(re.sub(r'\D', '', coef))
                if coef not in ('-', '+')
                else sign_map.get(coef)
                for coef in coefs
            ]

            for x_id, coef in zip(x_idx, coefs):
                matrix[row_id][x_id] = coef

            # Условие для отрицательного b
            if b[row_id] <= 0:
                matrix[row_id] = matrix[row_id] * -1
                b[row_id] = -b[row_id]

        matrix = np.array(matrix)

        # Добавляем переменные там, где знак неравенства
        for row_id, sign in enumerate(row_sign):
            if sign == '<=':
                x_a = np.array([[0 for _ in range(len(b))]])
                x_a[0][row_id] = 1
                matrix = np.concatenate([matrix, x_a.T], axis=1)
            elif sign == '>=':
                x_a = np.array([[0 for _ in range(len(b))]])
                x_a[0][row_id] = -1
                matrix = np.concatenate([matrix, x_a.T], axis=1)

        # Список переменных целевых и переменных матрицы
        x_matrix = [i for i in range(matrix.shape[1])]
        x_target = [matrix.shape[1] + i for i in range(len(b))]

        return matrix, b, x_matrix, x_target

    def step_algo(self):
        q = -np.sum(self._A, axis=0).reshape(1, -1)
        b = np.append(self._b, np.array([-np.sum(self._b, axis=0)]), axis=0)
        b = np.reshape(b, (len(b), 1))

        # Начало итерационного алгоритма
        target_col_idx = np.argmax(np.abs(q))
        a_div = [x if x > 0 else np.maximum for x in b / self._A[:, target_col_idx]]
        target_row_idx = np.argmin(a_div)

        # Замена переменных
        self._x_matrix[target_col_idx], self._x_target[target_row_idx] = self._x_target[target_row_idx], self._x_matrix[target_col_idx]

        # Добавляем доп столбцы в основную матрицу
        prev_matrix = deepcopy(self._A)
        prev_matrix = np.append(prev_matrix, q, axis=0)
        prev_matrix = np.concatenate((prev_matrix, b), axis=1)

        new_matrix = deepcopy(prev_matrix)

        # Расчет разрешающего элемента
        target_elem = self._A[target_row_idx, target_col_idx]

        q[target_col_idx] = q[target_col_idx] / target_elem
        self._b[target_row_idx] = self._b[target_row_idx] / target_elem

        # Изменение матрицы по формуле
        for row_idx in range(prev_matrix.shape[0]):
            for col_idx in range(prev_matrix.shape[1]):
                if target_col_idx == col_idx and target_row_idx == row_idx:
                    new_matrix[row_idx][col_idx] = 1 / target_elem
                elif row_idx == target_row_idx and target_col_idx != col_idx:
                    new_matrix[row_idx][col_idx] = prev_matrix[row_idx][col_idx] / target_elem
                elif col_idx == target_col_idx and row_idx != target_row_idx:
                    new_matrix[row_idx][col_idx] = -prev_matrix[row_idx][col_idx] / target_elem
                else:
                    new_matrix[row_idx][col_idx] -= (prev_matrix[target_row_idx][col_idx] * prev_matrix[row_idx][target_col_idx]) / target_elem

        print(new_matrix)


    @property
    def A(self):
        return self._A

    @property
    def b(self):
        return self._b

    @staticmethod
    def _find_max_x(matrix_rows: list) -> int:
        max_x = -1
        for row in matrix_rows:
            all_x = re.findall('x_[0-9]+', row)
            all_x = [int(x.replace('x_', '')) for x in all_x]
            if (curr_max := max(all_x)) > max_x:
                max_x = curr_max

        return max_x
