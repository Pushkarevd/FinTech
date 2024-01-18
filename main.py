import pandas as pd
import numpy as np


class DecisionMaking:
    def __init__(self, matrix: np.array, p: list, alpha: float):
        self._matrix = matrix
        self._p = p
        self._alpha = alpha

    def calculate_criteria(self):
        bayes = np.argmax(np.matmul(self._matrix, self._p))
        laplace = np.argmax(np.matmul(self._matrix, [1 / len(self._p)] * len(self._p)))
        wald = np.argmax(np.min(self._matrix, axis=1))
        savage = np.argmin(np.max(np.max(self._matrix, axis=0) - self._matrix, axis=1))
        hurwitz = np.argmax(
            self._alpha * np.min(self._matrix, axis=1)
            + (1 - self._alpha) * np.max(self._matrix, axis=1)
        )

        return pd.DataFrame(
            [[bayes, laplace, wald, savage, hurwitz]],
            columns=["Байес", "Лаплас", "Вальд", "Севидж", "Гурвиц"],
            index=["Критерий"],
        )


if __name__ == "__main__":
    matrix = np.array(
        [
            [2.3, 4.1, 3.5, 6.7, 8.9],
            [10.2, 5.6, 2.1, 6.4, 7.8],
            [3.4, 6.7, 1.2, 9.8, 9.5],
            [4.5, 7.8, 2.3, 10.1, 10.2],
            [5.6, 8.9, 3.4, 11.2, 11.3],
            [6.7, 10, 4.5, 12.3, 12.4],
            [7.8, 11.1, 5.6, 13.4, 13.5],
            [8.9, 12.2, 6.7, 14.5, 14.6],
            [10, 13.3, 7.8, 15.6, 15.7],
            [11.1, 14.4, 8.9, 16.7, 16.8],
            [12.2, 15.5, 10, 17.8, 17.9],
            [13.3, 16.6, 11.1, 18.9, 19],
            [14.4, 17.7, 12.2, 20, 20.1],
        ]
    )

    instance = DecisionMaking(matrix, [0.2, 0.1, 0.3, 0.2, 0.2], 0.4)
    print(instance.calculate_criteria())
