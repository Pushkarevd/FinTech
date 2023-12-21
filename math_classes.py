import re
from copy import deepcopy
from fractions import Fraction

import numpy as np
import pandas as pd

pd.set_option("display.precision", 2)


class TargetFunc:

    def __init__(self, func_string: str):
        func, target = func_string.split()
        self._func = func.strip()
        self._optimization_task = target.strip()

        self._func_matrix = self._target_matrix()

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
            matrix[x_id] = coef

        matrix[-1] = bias

        matrix = np.array([float(x) if x is not None else 0 for x in matrix])
        if self.target == 'max':
            matrix = -matrix
        return matrix

    @property
    def func(self):
        return self._func

    @property
    def target(self):
        return self._optimization_task

    @property
    def func_matrix(self):
        return self._func_matrix

    @property
    def result_matrix(self):
        if self.target == 'max':
            return -self._func_matrix
        else:
            return self._func_matrix


class Task:

    def __init__(self, target_task: TargetFunc, matrix_string: str):
        self._target_task = target_task
        self._A, self._b, self._x_columns, self._x_rows = self._prepare_matrix(matrix_string)

        self._x_aditional_history = deepcopy(self._x_rows)

    def _prepare_matrix(self, matrix_string: str) -> tuple:
        """
        Функция для предобработки входных данных и формирования матрицы А и b
        :param matrix_string: входные данные, список ограничений
        :return: A, b матрицы
        """
        # Разделение строки на A и b
        prepared_strings = matrix_string.replace(' ', '') \
                               .split('\n')[1:]

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
        matrix = matrix + Fraction()

        self._x_variables_hist = [x for x in range(matrix.shape[0])]

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

    def __debug_print(self, matrix: np.array):
        tmp_columns = [x + 1 for x in self._x_columns]
        tmp_index = [x + 1 for x in self._x_rows]
        tmp_columns.append('b')
        tmp_index.append('c')
        print(pd.DataFrame(matrix, columns=tmp_columns, index=tmp_index))

    def _simplex_method(self, matrix: np.array) -> np.array:
        # Пока последняя строка матрицы вся не будет состоять из неотрицательных элементов
        while not all([round(x, 4) >= 0 for x in matrix[-1, :]]):
            self.__debug_print(matrix)
            # Определение разрешающего столбца
            target_col = np.argmin([round(x, 4) for x in matrix[-1, :-1]])

            # Определение разрешающей строки
            b = matrix[:-1, -1]
            target_col_elem = matrix[:-1, target_col]
            target_row = np.argmin([b_x / m_x if m_x > 0 else 10 ** 10 for m_x, b_x in zip(target_col_elem, b)])

            # Замена переменных строк и столбца
            self._x_columns[target_col], self._x_rows[target_row] = self._x_rows[target_row], self._x_columns[
                target_col]

            # Формируем матрицу следующей итерации
            next_matrix = deepcopy(matrix)

            # Расчет следующей итерации
            for row_idx in range(matrix.shape[0]):
                for col_idx in range(matrix.shape[1]):
                    if target_row == row_idx and target_col == col_idx:
                        next_matrix[target_row][target_col] = 1 / matrix[target_row][target_col]
                    elif target_col == col_idx and target_row != row_idx:
                        next_matrix[row_idx][col_idx] = -matrix[row_idx][col_idx] / matrix[target_row][target_col]
                    elif target_row == row_idx and target_col != col_idx:
                        next_matrix[row_idx][col_idx] = matrix[row_idx][col_idx] / matrix[target_row][target_col]
                    else:
                        next_matrix[row_idx][col_idx] = (matrix[row_idx][col_idx]
                                                         - ((matrix[target_row][col_idx] * matrix[row_idx][target_col])
                                                            / matrix[target_row][target_col]))

            # Удаление столбца с вспомогательной переменной
            check_aditional = [x for x in self._x_columns if x in self._x_aditional_history]
            if len(check_aditional) > 0:
                add_idx = self._x_columns.index(check_aditional[0])
                next_matrix = np.delete(next_matrix, add_idx, axis=1)
                self._x_columns.remove(check_aditional[0])

            matrix = next_matrix
        return matrix

    def additional_step(self):
        matrix = np.append(self._A, self._b.reshape(1, -1).T, axis=1)
        last_row = -np.sum(matrix, axis=0).reshape(1, -1)
        matrix = np.append(matrix, last_row, axis=0)

        print('Формирование изначальной матрицы')
        self.__debug_print(matrix)

        return self._simplex_method(matrix)

    def get_new_function(self, matrix: np.array) -> np.array:
        vectors = []

        for var in self._x_variables_hist:
            if var in self._x_rows:
                idx = self._x_rows.index(var)
                x_vector = np.append(-matrix[idx, :-1], matrix[idx, -1])
                vector = x_vector * self._target_task.func_matrix[var]
            else:
                idx = self._x_columns.index(var)
                vector = np.array([0 for _ in range(len(self._x_columns) + 1)])
                vector[idx] = 1
                vector = np.array(vector) + Fraction()
            vectors.append(vector)

        new_function = np.sum(np.array(vectors), axis=0)
        new_function[-1] = -new_function[-1]

        return new_function

    def get_result(self, matrix: np.array):
        vector = [0 for _ in range(len(self._x_rows) + len(self._x_columns))]
        for x, val in zip(self._x_rows, matrix[:-1, -1]):
            vector[x] = val

        print('Результирующий вектор')
        print([round(x, 4) for x in vector])

        result = np.sum(
            [
                t_x * coef
                for t_x, coef in
                zip(vector[:len(self._target_task.result_matrix[:-1])], self._target_task.result_matrix[:-1])
            ]
        ) + self._target_task.result_matrix[-1]

        print(f'Значение функции в точке:')
        print(round(result, 4))

    def algo(self):
        matrix = self.additional_step()
        print('Окончание выполнения вспомогательной задачи')
        self.__debug_print(matrix)

        new_function = self.get_new_function(matrix)
        target_matrix = deepcopy(matrix)
        target_matrix[-1] = new_function
        matrix = self._simplex_method(target_matrix)
        self.__debug_print(matrix)
        self.get_result(matrix)

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
