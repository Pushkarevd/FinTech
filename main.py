import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import yfinance


class HoltWinters:

    def __init__(self, series: pd.Series, L: int, alphas: [float, float, float]):
        self.series = series
        self._default_index = self.series.index
        self.series.index = range(1, self.series.shape[0] + 1)

        self._test_series = series.tail(L)
        self.series = self.series.iloc[:series.shape[0] - L + 1]

        self._l = L
        self._alpha_1, self._alpha_2, self._alpha_3 = alphas

    def init_model(self) -> tuple:
        mean_x = np.mean(self.series.index)
        mean_y = np.mean(self.series)

        numerator = np.sum((self.series.index - mean_x) * (self.series - mean_y))
        denominator = np.sum((self.series.index - mean_x) ** 2)

        a = numerator / denominator
        b = mean_y - a * mean_x

        predictions = [a * i + b for i in self.series.index]
        diff = [fact / pred for fact, pred in zip(self.series, predictions)]

        k = self.series.shape[0] // self._l
        f = [sum(diff[i:self.series.shape[0]:self._l]) / k for i in range(k + 1)]

        return a, b, f

    def algo(self):

        A, B, TAU = [], [], []

        a, b, seasonals = self.init_model()
        A.append(a)
        B.append(b)

        for idx in range(self.series.shape[0]):
            b = self._alpha_1 * (self.series.iloc[idx] / seasonals[idx]) + (1 - self._alpha_1) * (B[idx] + A[idx])
            B.append(b)

            a = self._alpha_2 * (B[idx + 1] - B[idx]) + (1 - self._alpha_2) * A[idx]
            A.append(a)

            season = self._alpha_3 * self.series.iloc[idx] / b + (1 - self._alpha_3)*seasonals[idx]

            seasonals.append(season)
            tau = ((idx % self._l + 1) * A[idx // self._l * self._l] + B[idx // self._l * self._l]) * seasonals[idx]
            TAU.append(tau)

        predictions = []
        # Предсказание тестовой выборки
        for idx in range(self._l):
            pred = ((idx % self._l + 1) * A[-1] + B[-1]) * seasonals[len(seasonals) - (self._l - idx)]
            predictions.append(pred)

        TAU = TAU[:-1]
        TAU.extend(predictions)

        self.plot_result(TAU, predictions)
        self.calc_metric(predictions)

    def plot_result(self, predictions: list, test_predictions: list):
        sns.set_theme(style="darkgrid")
        sns.lineplot(x=self._default_index, y=predictions, label='Прогноз')
        sns.lineplot(x=self._default_index[:self.series.shape[0]], y=self.series, label='Фактические данные')
        sns.lineplot(x=self._default_index[-self._l:], y=self._test_series, label='Тестовые данные')
        sns.lineplot(x=self._default_index[-self._l:], y=test_predictions, label='Прогноз тестовых данных')
        plt.xlabel('Дата')
        plt.show()

    def calc_metric(self, predictions: list):
        mse = sum([(tr - pr) ** 2 for tr, pr in zip(self._test_series, predictions)]) / len(predictions)
        print(f'MSE - {mse:.4f}')


if __name__ == '__main__':
    df = yfinance.Ticker('TSLA').history(start="2023-06-01", end=None)['Close']
    #df = pd.Series([49, 78, 84, 94, 65, 99, 111, 142, 103, 189, 152, 181])
    instance = HoltWinters(df, 7, [0.4, 0.3, 0.6])
    instance.algo()
