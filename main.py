import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN


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

    def __kendall(self):
        m = self._matrix.shape[1]
        n = self._matrix.shape[0]

        sums = [sum(x) for x in self._matrix]
        Rbar = sum(sums) / n
        S = sum([(sums[x] - Rbar) ** 2 for x in range(n)])

        W = (12 * S) / (m ** 2 * (n ** 3 - n))

        return W

    def __write_result(self):
        with open("output.txt", "w") as file:
            file.write(
                f'Вектор оценки экспертов: {self.__get_objects_estimation().values}\n \
                Вектор квалификации экспертов: {self.__get_expert_qualification().values}'
            )

    def __plot_dots(self):
        pca = PCA(n_components=2)

        pca.fit(self._matrix)
        X_pca = pca.transform(self._matrix)
        clusters = self.__cluster()

        plt.title('Визуализация с методом PCA и кластеризации DBSCAN')
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters)

        plt.grid()
        plt.show()

    def __cluster(self):
        clf = DBSCAN(min_samples=2, eps=2)
        clusters = clf.fit_predict(self._matrix)
        return clusters

    def start(self):
        print(self.matrix)
        self.__write_result()
        print(f'Kendall W - {self.__kendall()}')
        self.__plot_dots()


if __name__ == "__main__":
    input_data = [row.split() for row in open("input.txt").read().split("\n")]
    input_data = [[float(x) for x in row] for row in input_data]

    matrix = np.array(input_data)
    instance = ExpertTask(matrix)
    instance.start()
