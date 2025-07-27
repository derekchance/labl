from pathlib import Path
import pandas as pd
import numpy as np
from joblib import load

from guts import determine_linear_weights

year = 2025


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


def main():
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

    off_df['wOBA'] = ((off_df.loc[:, ['1B', '2B', '3B', 'HR', 'BB', 'HBP']] * lw_df.loc[['1B', '2B', '3B', 'HR', 'BB', 'HBP']]).sum(axis=1)
                      / (off_df.loc[:, ['AB', 'BB', 'HBP', 'SF']].sum(axis=1))
                      )
    off_df['wRC'] = (((off_df['wOBA'] - lg_off_df['OBP']) / wOBA_scale) + (lg_rpa)) * off_df['PA']
    off_df['wRAA'] = ((off_df['wOBA'] - lg_off_df['OBP'])/ wOBA_scale) * off_df['PA']
    off_df['wRC+'] = 100 * (off_df['wRAA']/off_df['PA'] + lg_rpa) / (off_df.wRC.sum() / off_df.PA.sum())
    off_df['wSB'] = off_df['SB'] * 0.2 - lgwSB * (off_df['1B'] + off_df['BB'] + off_df['HBP'])
    off_df['Off'] = off_df['wRAA'] + off_df['wSB']
    off_df['Rep'] = (2*RPW / 600) * off_df['PA']
    off_df['RAR'] = off_df['Off'] + off_df['Rep']
    off_df['WAR'] = off_df['RAR'] / RPW

    off_df.sort_values('WAR', ascending=False).fillna('').to_csv('results/2025/offense.csv', index=False)


if __name__ == '__main__':
    main()
