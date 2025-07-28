import pandas as pd

from standings import main as standings
from hitters import main as hitters
from pitchers import main as pitchers

YEAR = 2025

def main():
    standings()
    hitters()
    pitchers()

    off_df = pd.read_csv(f'results/{YEAR}/offense.csv')
    pitch_df = pd.read_csv(f'results/{YEAR}/pitching.csv')
    war_df = pd.merge(
        off_df[['Name', 'Team', 'WAR']],
        pitch_df[['Name', 'Team', 'WAR']],
        how='outer',
        on=['Name', 'Team'],
        suffixes=('_off', '_RA9')
    )
    war_df['WAR'] = war_df['WAR_off'].fillna(0) + war_df['WAR_RA9'].fillna(0)
    war_df.sort_values('WAR', ascending=False).fillna('').to_csv('results/2025/war.csv', index=False)


if __name__ == '__main__':
    main()
