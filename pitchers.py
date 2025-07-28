from pathlib import Path
import pandas as pd

year = 2025


def load_pitching_data():
    pitch_path = Path('./data/pitching/2025')
    pitch_df = pd.concat([pd.read_csv(n) for n in pitch_path.glob('*.csv')])
    pitch_df['IP_dec'] = pitch_df['IP'].round() + (pitch_df['IP'] % 1 * (10/3))
    return pitch_df


def main():
    pitch_df = load_pitching_data()
    lg_RA9 = 9 * (pitch_df.R.sum() / pitch_df.IP_dec.sum())
    pitch_df['IP_p_G'] = pitch_df['IP_dec'] / pitch_df['GP']
    pitch_df['RA9'] = 9 * pitch_df['R'] / pitch_df['IP_dec']
    pitch_df.dropna(how='any', inplace=True)
    pitch_df['dRPW'] = (((((18 - pitch_df['IP_p_G']) * lg_RA9) + (pitch_df['IP_p_G']*pitch_df['RA9'])) / 18) + 2)*1.5
    pitch_df['RAAP9'] = lg_RA9 - pitch_df['RA9']
    pitch_df['WPGAA'] = pitch_df['RAAP9'] / pitch_df['dRPW']
    pitch_df['Rep'] = 0.11
    pitch_df['WPGAR'] = pitch_df['WPGAA'] + pitch_df['Rep']
    pitch_df['WAR'] = pitch_df['WPGAR'] * (pitch_df['IP_dec'] / 9)

    pitch_df['Name'] = pitch_df.Name.str.replace('\xa0', ' ')
    pitch_df.sort_values('WAR', ascending=False).fillna('')\
        .to_csv('results/2025/pitching.csv', index=False)



if __name__ == '__main__':
    main()
