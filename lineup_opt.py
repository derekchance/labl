import pandas as pd
import numpy as np
from scipy.stats import poisson, beta
import matplotlib.pyplot as plt
from itertools import permutations

YEAR_SCOPE = [2024, 2025]
PLAYOFF_ELIG = 7
INIT_HYPERPARAMETERS = {
    '1B': (3, 15),
    '2B': (2, 47.2),
    '3B': (2, 621.6),
    'HR': (0.5, 47.4),
    'BB': (2, 21.5),
    'HBP': (1.5, 50.55),
    'K': (4, 18.45),
}

class Player:
    def __init__(self, player_name):
        self.player_name = player_name
        self.hyperparams = INIT_HYPERPARAMETERS
        self.weights = {}

    def ab(self):
        return np.random.choice(list(self.weights.keys()), p=list(self.weights.values()))

    def update_hyperparams(self, results):
        pa = results.loc['PA']
        weights = {}
        for i in ['1B', '2B', '3B', 'HR', 'BB', 'K', 'HBP']:
            a, b = self.hyperparams[i]
            a += results.loc[i]
            b += (pa - a)
            self.hyperparams[i] = a, b
            self.weights[i] = beta.mean(a, b)

        nonk = 1 - np.sum(list(self.weights.values()))
        self.weights['OUT'] = nonk * .95
        self.weights['ERR'] = nonk * .05


class Lineup:
    def __init__(self, order: tuple, weights: pd.DataFrame):
        self.order = order
        self.length = len(order)
        self.index = 0
        self.weights = weights

    def __iter__(self):
        return self

    def __next__(self):
        if self.index == self.length:
            self.index = 0
        player = self.order[self.index]
        self.index += 1
        return np.random.choice(self.weights.columns, p=self.weights.loc[player])


class Game(Lineup):
    def __init__(self, order: tuple, weights: pd.DataFrame):
        inning = 1
        outs = 0
        runs = 0
        super().__init__(order, weights)



def main():
    df = pd.concat({n: pd.read_csv(f'results/{n}/offense.csv') for n in YEAR_SCOPE})
    sum_df = df.groupby('Name')[['PA', '1B', '2B', '3B', 'HR', 'BB', 'K', 'HBP']].sum()
    l_df = df.set_index('Team', append=True).loc[(2025, slice(None), 'Legends')]
    playoff_eligible = l_df.loc[l_df.GP >= PLAYOFF_ELIG, 'Name'].values
    lu_df = sum_df.loc[playoff_eligible]
    weights_df = lu_df.copy()

    lw_df = pd.read_csv('data/guts/linear_weights.csv', index_col='outcome')
    bayes_woba = (weights_df['1B'] * lw_df.loc['1B', 'run_exp_change']
                  + weights_df['2B'] * lw_df.loc['2B', 'run_exp_change']
                  + weights_df['3B'] * lw_df.loc['3B', 'run_exp_change']
                  + weights_df['HR'] * lw_df.loc['HR', 'run_exp_change']
                  + weights_df['BB'] * lw_df.loc['BB', 'run_exp_change']
                  + weights_df['HBP'] * lw_df.loc['HBP', 'run_exp_change']
                  + weights_df['ERR'] * lw_df.loc['1B', 'run_exp_change']
                  ).sort_values(ascending=False)