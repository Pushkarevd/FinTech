import pandas as pd
import numpy as np

# from sklearn.linear_model import LinearRegression


class ExpertTask:
    def __init__(self, matrix: np.array):
        self._matrix = matrix

    @property
    def matrix(self):
        df = pd.DataFrame(
            self._matrix,
            columns=[f"Эксперт {i + 1}" for i in range(self._matrix.shape[1])],
            index=[f"Объект {i + 1}" for i in range(self._matrix.shape[0])],
        )
        return df

    def __get_objects_estimation(self):
        B_matrix = self._matrix @ self._matrix.T
        vect_y = [
            (B_matrix[i].prod()) ** (1 / B_matrix.shape[1])
            for i in range(B_matrix.shape[0])
        ]
        return pd.Series(
            [yl / sum(vect_y) for yl in vect_y],
            index=[i + 1 for i in range(B_matrix.shape[0])],
        )

    def __get_expert_qualification(self):
        C_matrix = self._matrix.T @ self._matrix
        vect_y = [
            (C_matrix[i].prod()) ** (1 / C_matrix.shape[1])
            for i in range(C_matrix.shape[0])
        ]
        return pd.Series(
            [yl / sum(vect_y) for yl in vect_y],
            index=[i + 1 for i in range(C_matrix.shape[1])],
        )

    def __write_result(self):
        with open("output.txt", "w") as file:
            file.write(
                f'Вектор оценки экспертов: {self.__get_objects_estimation().values}\nВектор квалификации экспертов: {self.__get_expert_qualification().values}'
            )

    def start(self):
        print(self.matrix)
        self.__write_result()


if __name__ == "__main__":
    input_data = [row.split() for row in open("input.txt").read().split("\n")]
    input_data = [[float(x) for x in row] for row in input_data]

    matrix = np.array(input_data)
    instance = ExpertTask(matrix)
    instance.start()
