import pandas as pd
import numpy as np
from scipy.optimize import minimize
from gekko import GEKKO
from tqdm import tqdm


class InvestOptimizer:

    def __init__(self, df: pd.DataFrame, cash: float):
        self._df = df
        self._cash = cash

        self._calc_metrics()
        self.laplace()

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

    def baies(self):
        expected_values = np.array(self._metric_df['ev'])
        init_prices = np.array(self._df['init_price'])

        max_shares = (self._cash / init_prices).astype(int)

        m = GEKKO(remote=False)
        shares = [m.Var(lb=0, ub=max_price, integer=True) for max_price in max_shares]

        m.Equation(m.sum([share * price for share, price in zip(shares, init_prices)]) <= self._cash)
        m.Obj(-m.sum([share * ev for share, ev in zip(shares, expected_values)]))

        m.options.SOLVER = 1
        m.solve(disp=False)

        for ticker, share in zip(self._df.index, shares):
            print(f'Количество акций {ticker}: {int(share.value[0])}')

    def laplace(self):
        columns = df.columns.values
        prob_columns = np.delete(columns, np.argwhere(columns == 'init_price'))

        average_returns = self._df[prob_columns].mean(axis=1)
        print(average_returns)

    def vald(self):
        pass

if __name__ == "__main__":
    cash = 500_000
    df = pd.read_json('data.json').T
    instance = InvestOptimizer(df, cash)
