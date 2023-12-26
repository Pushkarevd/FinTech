import pandas as pd
import numpy as np
from scipy.optimize import minimize


class InvestOptimizer:

    def __init__(self, df: pd.DataFrame, cash: float):
        self._df = df
        self._cash = cash

        self._calc_metrics()
        self.baies()

    def _calc_metrics(self):
        columns = df.columns.values
        prob_columns = np.delete(columns, np.argwhere(columns == 'init_price'))
        result = []
        for index, row in self._df.iterrows():
            k = [(row[prob] - row['init_price']) / row['init_price'] for prob in prob_columns]
            m = sum([k_i * prob for k_i, prob in zip(k, row[prob_columns])])
            disp = sum([(k_i * m) ** 2 * prob for k_i, prob in zip(k, row[prob_columns])])
            sigma = disp ** 0.5
            cv = sigma / m

            k.extend([m, disp, sigma, cv])

            result.append(k)

        result_columns = [f'Z_{i}' for i in range(1, len(prob_columns) + 1)]
        result_columns.extend(['ev', 'disp', 'sigma', 'cv'])

        self._metric_df = pd.DataFrame(result, columns=result_columns)
        print(self._metric_df)

    def baies(self):
        expected_values = np.array(self._metric_df['ev'])
        init_prices = np.array(self._df['init_price'])

        n = df.shape[0]

        loss = lambda x: -np.sum(expected_values * x)

        constraints = (
            {"type": "ineq", "fun": lambda x: np.sum(init_prices * x) - self._cash}
        )

        x0 = np.full(n, 1 / n)

        result = minimize(loss, x0, constraints=constraints)

        optimal_investments = result.x

        print("Optimal investments:", ', '.join([f'{x:.2f}' for x in optimal_investments]))
        print(self._metric_df['ev'])

if __name__ == "__main__":
    cash = 50_000
    df = pd.read_json('data.json').T
    instance = InvestOptimizer(df, cash)
