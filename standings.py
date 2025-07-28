import pandas as pd

year = 2025


def main():
    global year
    score_df = pd.read_csv(f'data/scores/{year}.csv')
    score_df['visitor_score'] = score_df['Score'].str.split('-').str[0].astype(int)
    score_df['home_score'] = score_df['Score'].str.split('-').str[1].astype(int)
    rs_df = score_df.groupby('Visitors').visitor_score.sum() + score_df.groupby('Home').home_score.sum()
    ra_df = score_df.groupby('Visitors').home_score.sum() + score_df.groupby('Home').visitor_score.sum()
    runs_df = pd.concat([rs_df, ra_df], axis=1, keys=['Runs Scored', 'Runs Allowed'])
    score_df['home_win'] = (score_df.home_score > score_df.visitor_score)
    score_df['home_loss'] = (score_df.home_score < score_df.visitor_score)
    score_df['home_tie'] = (score_df.home_score == score_df.visitor_score)
    score_df['visitor_win'] = score_df['home_loss']
    score_df['visitor_loss'] = score_df['home_win']
    score_df['visitor_tie'] = score_df['home_tie']
    record_df = (
            score_df.groupby('Home')[['home_win', 'home_loss', 'home_tie']].sum()
            .rename(columns=lambda x: x.replace('home_',''))
            + score_df.groupby('Visitors')[['visitor_win', 'visitor_loss', 'visitor_tie']].sum()
            .rename(columns=lambda x: x.replace('visitor_',''))
    )

    standings_df = pd.concat([record_df, runs_df], axis=1)
    standings_df['G'] = standings_df[['win', 'loss', 'tie']].sum(axis=1)
    pythag = ((standings_df['Runs Scored'] ** 1.83)
              / (standings_df['Runs Scored'] ** 1.83 + standings_df['Runs Allowed'] ** 1.83))
    standings_df['Pythag W'] = (pythag * standings_df['G']).round(1)
    standings_df['Pythag L'] = 15 - standings_df['Pythag W']
    standings_df.index = standings_df.index.str.strip()
    standings_df.index.name = 'Team'
    standings_df.to_csv(f'data/standings/{year}.csv')


if __name__ == '__main__':
    main()
