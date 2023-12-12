import re
from copy import deepcopy

import numpy as np


class TargetFunc:

    def __init__(self, func_string: str):
        func, target = func_string.split()
        self._func = func.strip()
        self._optimization_task = target.strip()

        self.func_matrix = self._target_matrix()

    @staticmethod
    def _find_max_x(func: str) -> int:
        max_x = -1
        all_x = re.findall('x_[0-9]+', func)
        all_x = [int(x.replace('x_', '')) for x in all_x]
        if (curr_max := max(all_x)) > max_x:
            max_x = curr_max

        return max_x

    def _target_matrix(self):
        matrix = [0 for _ in range(self._find_max_x(self._func) + 1)]
        sign_map = {'-': -1, '+': 1}

        bias = re.search('[0-9]+[^x\+-]', self._func)
        if bias:
            bias = bias.group(0)

        x_idx = [int(x) - 1 for x in re.findall(r'x_(\d+)', self._func)]
        coefs = re.split('x_[0-9]+', self._func)[:-1]

        coefs = [
            float(re.sub(r'\D', '', coef))
            if coef not in ('-', '+')
            else sign_map.get(coef)
            for coef in coefs
        ]

        for x_id, coef in zip(x_idx, coefs):
            matrix[x_id + 1] = coef

        matrix[0] = bias

        matrix = np.array([float(x) if x is not None else 0 for x in matrix])
        return matrix

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

        self._x_target_hist = deepcopy(self._x_target)

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

        self._x_matrix_hist = [x for x in range(matrix.shape[0])]

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

    def algo(self):
        prev_matrix = deepcopy(self._A)
        b = deepcopy(self._b)
        q = -np.sum(self._A, axis=0).reshape(1, -1)
        prev_matrix = np.append(prev_matrix, q, axis=0)
        b = np.append(b, np.array([-np.sum(b, axis=0)]), axis=0)
        b = np.reshape(b, (len(b), 1))
        prev_matrix = np.concatenate((prev_matrix, b), axis=1)

        # Начало итерационного алгоритма
        while not all([round(x, 3) == 0 for x in prev_matrix[-1, :]]):
            trg_col_idx = []
            for col in self._x_matrix_hist:
                if col in self._x_matrix:
                    trg_col_idx.append(self._x_matrix.index(col))

            target_col_idx = np.argmin(prev_matrix[-1, trg_col_idx])
            a_div = [x if x > 0 else 10 ** 10 for x in prev_matrix[:-1, -1] / prev_matrix[:-1, target_col_idx]]
            target_row_idx = np.argmin(a_div)

            # Замена переменных
            self._x_matrix[target_col_idx], self._x_target[target_row_idx] = self._x_target[target_row_idx], \
                self._x_matrix[target_col_idx]

            # Добавляем доп столбцы в основную матрицу
            new_matrix = deepcopy(prev_matrix)

            # Расчет разрешающего элемента
            target_elem = prev_matrix[target_row_idx, target_col_idx]

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
                        new_matrix[row_idx][col_idx] -= (prev_matrix[target_row_idx][col_idx] * prev_matrix[row_idx][
                            target_col_idx]) / target_elem

            check = [x for x in self._x_matrix if x in self._x_target_hist]
            if len(check) > 0:
                add_idx = self._x_matrix.index(check[0])
                prev_matrix = np.delete(new_matrix, add_idx, axis=1)
                self._x_matrix.remove(check[0])

        target_x = [x for x in self._x_target if x in self._x_matrix_hist]

        x_mapping = {}

        for x_idx, x_var in enumerate(target_x):
            x_coefs = -prev_matrix[x_idx, :-1]
            bias = prev_matrix[x_idx, -1]
            x_mapping |= {x_var: np.append(bias, x_coefs)}

        target_matrix = self._target_task.func_matrix

        print(target_matrix)
        print(x_mapping)

        for x_idx, coef in enumerate(target_matrix[1:]):
            print(coef, x_idx)

        z_matrix = np.sum([
            x_mapping[x_idx] * coef
            for x_idx, coef in enumerate(target_matrix[1:])
        ], axis=0)

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
