from pathlib import Path
import pandas as pd
import numpy as np
from joblib import load

from guts import determine_linear_weights

year = 2025

POS_ADJ = {
    'P': 0,
    'C': 10.75,
    '1B': -11,
    '2B': 2.75,
    '3B': 2.25,
    'SS': 7.25,
    'CF': 2.5,
    'COF': -7.25,
    'EH': -11.5,
}


def load_offense_data():
    global year
    off_path = Path(f'./data/offense/{year}')
    off_df = pd.concat([pd.read_csv(n) for n in off_path.glob('*.csv')])
    off_df['1B'] = off_df['H'] - off_df['2B'] - off_df['3B'] - off_df['HR']
    off_df['PA'] = off_df['AB'] + off_df['BB'] + off_df['HBP'] + off_df['SH'] + off_df['SF']

    lg_off_df = off_df.loc[:, 'GP': 'TB'].sum()
    lg_off_df['PA'] = lg_off_df['AB'] + lg_off_df['BB'] + lg_off_df['HBP'] + lg_off_df['SH'] + lg_off_df['SF']
    lg_off_df['1B'] = off_df['1B'].sum()
    lg_off_df['OBP'] = (lg_off_df['H'] + lg_off_df['BB'] + lg_off_df['HBP']) / lg_off_df['PA']
    return off_df, lg_off_df


def calc_owar():
    off_df, lg_off_df = load_offense_data()
    lw_df = determine_linear_weights(lg_off_df)

    lg_rpa = lg_off_df.R / lg_off_df.PA
    lgwSB = (lg_off_df.SB * 0.2) / (lg_off_df['1B'] + lg_off_df['BB'] + lg_off_df['HBP'])

    wOBA_scale = load(Path('./data/guts/wOBA_scale.joblib'))
    try:
        RPW = load(Path('./data/guts/RPW.joblib'))
    except FileNotFoundError:
        from guts import determine_rpw
        RPW = determine_rpw()

    off_df['wOBA'] = ((off_df.loc[:, ['1B', '2B', '3B', 'HR', 'BB', 'HBP']]
                       * lw_df.loc[['1B', '2B', '3B', 'HR', 'BB', 'HBP']]).sum(axis=1)
                      / (off_df.loc[:, ['AB', 'BB', 'HBP', 'SF']].sum(axis=1))
                      )
    off_df['wRC'] = (((off_df['wOBA'] - lg_off_df['OBP']) / wOBA_scale) + (lg_rpa)) * off_df['PA']
    off_df['wRAA'] = ((off_df['wOBA'] - lg_off_df['OBP'])/ wOBA_scale) * off_df['PA']
    off_df['wRC+'] = 100 * (off_df['wRAA']/off_df['PA'] + lg_rpa) / (off_df.wRC.sum() / off_df.PA.sum())
    off_df['wSB'] = off_df['SB'] * 0.2 - lgwSB * (off_df['1B'] + off_df['BB'] + off_df['HBP'])
    off_df['Off'] = off_df['wRAA'] + off_df['wSB']
    off_df['Rep'] = (2*RPW / 600) * off_df['PA']
    off_df['RAR'] = off_df['Off'] + off_df['Rep']
    off_df['oWAR'] = off_df['RAR'] / RPW

    return off_df


def calc_dwar(off_df):
    def_path = Path(f'./data/defense/{year}')
    def_df = pd.concat([pd.read_csv(n).set_index('Name').stack() for n in def_path.glob('*.csv')]).rename('share')
    def_df.index.names = ['Name', 'POS']
    no_EH = off_df.loc[off_df.Team == 'Legends', 'GP'].sum() / 14 - 8
    pos_adj = pd.read_csv(Path('data/guts/pos_adj.csv'), index_col='POS')
    pos_adj['fg'] /= 162
    pos_adj['br'] /= 150
    pos_adj['pos'] = pos_adj[['fg', 'br']].mean(axis=1)
    rep_level = pos_adj.loc[['1B', 'LF', 'RF'], 'pos'].mean()
    pos_adj['pos'] -= rep_level
    pos_adj.loc['C': 'RF', 'pos'] /= -1 * (pos_adj.loc['C': 'RF', 'pos'].sum() / (pos_adj.loc['EH', 'pos'] * no_EH))
    pos_adj.loc['P'] = [0, 0, 0]
    pos_adj.loc['COF'] = pos_adj.loc[['LF', 'RF']].mean()
    player_pos_adj = (def_df * pos_adj.pos).groupby(level=0).sum().rename('positional_adjustment').to_frame()
    player_pos_adj['Team'] = 'Legends'
    player_pos_adj.set_index('Team', append=True, inplace=True)
    off_df = off_df.merge(player_pos_adj, how='left', left_on=['Name', 'Team'], right_index=True)
    off_df['dRAR'] = off_df['GP'] * off_df['positional_adjustment']
    try:
        RPW = load(Path('./data/guts/RPW.joblib'))
    except FileNotFoundError:
        from guts import determine_rpw
        RPW = determine_rpw()
    off_df['dWAR'] = off_df['dRAR'] / RPW
    return off_df


def main():
    off_df = calc_owar()
    off_df = calc_dwar(off_df)

    off_df['WAR'] = off_df['oWAR'] + off_df['dWAR'].fillna(0.)
    off_df.sort_values('WAR', ascending=False).fillna('').to_csv('results/2025/offense.csv', index=False)


if __name__ == '__main__':
    main()
