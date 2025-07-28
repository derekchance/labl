from pathlib import Path
import pandas as pd
import numpy as np
from joblib import dump

from pitchers import load_pitching_data


def inning_sim(n, outcomes, verbose=0):
    outs = 0
    first = 0
    second = 0
    third = 0
    runs = 0
    innings = 0

    results = []
    while innings < n:
        b_outs = outs
        b_first = first
        b_second = second
        b_third = third
        b_runs = runs
        result = np.random.choice(outcomes)
        if result == 'OUT':
            err = np.random.rand()
            if err > 0.99:
                r = np.random.rand()
                if third == 1:
                    third = 0
                    runs += 1
                if second == 1:
                    runs += 1
                if first == 1:
                    first = 0
                    if r > 0.3:
                        runs += 1
                    else:
                        third = 1
                second = 1
            elif err > 0.94:
                r = np.random.rand()
                if third == 1:
                    third = 0
                    runs += 1
                if second == 1:
                    second = 0
                    if r > 0.3:
                        runs += 1
                    else:
                        third = 1
                if first == 1:
                    if third == 1:
                        second = 1
                    elif r > 0.7:
                        third = 1
                    else:
                        second = 1
                first = 1
            else:
                outs += 1
        elif result == '1B':
            r = np.random.rand()
            if third == 1:
                third = 0
                runs += 1
            if second == 1:
                second = 0
                if r > 0.3:
                    runs += 1
                else:
                    third = 1
            if first == 1:
                if third == 1:
                    second = 1
                elif r > 0.7:
                    third = 1
                else:
                    second = 1
            first = 1
        elif result == '2B':
            r = np.random.rand()
            if third == 1:
                third = 0
                runs += 1
            if second == 1:
                runs += 1
            if first == 1:
                first = 0
                if r > 0.3:
                    runs += 1
                else:
                    third = 1
            second = 1
        elif result == '3B':
            if third == 1:
                runs += 1
            if second == 1:
                second = 0
                runs += 1
            if first == 1:
                first = 0
                runs += 1
            third = 1
        elif result == 'HR':
            runs += 1 + first + second + third
            first = 0
            second = 0
            third = 0
        elif (result == 'BB') | (result == 'HBP'):
            if first == 1:
                if second == 1:
                    if third == 1:
                        runs += 1
                    third = 1
                second = 1
            else:
                first = 1
        results.append((result, b_outs, b_first, b_second, b_third, b_runs, outs, first, second, third, runs, innings))
        if outs > 2:
            outs = 0
            first = 0
            second = 0
            third = 0
            runs = 0
            innings += 1
        if verbose > 0:
            if innings % 1000 == 0:
                print(innings)
    return results


def get_run_exp(x, run_exp_df, i='b'):
    try:
        return run_exp_df.loc[(x[f'outs_{i}'], x[f'1B_{i}'], x[f'2B_{i}'], x[f'3B_{i}'])]
    except:
        return 0


def determine_linear_weights(lg_off_df):
    singles = ['1B' for n in np.arange(lg_off_df.loc['1B'])]
    doubles = ['2B' for n in np.arange(lg_off_df.loc['2B'])]
    triples = ['3B' for n in np.arange(lg_off_df.loc['3B'])]
    hrs = ['HR' for n in np.arange(lg_off_df.loc['HR'])]
    bb = ['BB' for n in np.arange(lg_off_df.loc['BB'])]
    hbp = ['HBP' for n in np.arange(lg_off_df.loc['HBP'])]
    sh = ['OUT' for n in np.arange(lg_off_df.loc['SH'])]
    sf = ['OUT' for n in np.arange(lg_off_df.loc['SF'])]

    outcomes = singles + doubles + triples + hrs + bb + hbp + sh + sf
    outcomes += ['OUT' for n in np.arange(lg_off_df.loc['PA'] - len(outcomes))]
    sim_df = pd.DataFrame.from_records(
        inning_sim(10000, outcomes=outcomes),
        columns=['outcome', 'outs_b', '1B_b', '2B_b', '3B_b', 'R_b', 'outs_a', '1B_a', '2B_a', '3B_a', 'R_a', 'Inning']
    )
    sim_df['R_inn'] = sim_df['Inning'].map(sim_df.groupby('Inning').R_a.max().to_dict())
    sim_df['R_rem'] = sim_df['R_inn'] - sim_df['R_b']

    run_exp_df = sim_df.groupby(['outs_b', '1B_b', '2B_b', '3B_b']).R_rem.mean()
    run_exp_df.unstack('outs_b').to_csv('data/guts/run_expectancy_matrix.csv')

    sim_df['run_exp_b'] = sim_df.apply(lambda x: get_run_exp(x, run_exp_df, i='b'), axis=1)
    sim_df['run_exp_a'] = sim_df.apply(lambda x: get_run_exp(x, run_exp_df, i='a'), axis=1)
    sim_df['run_exp_change'] = sim_df.run_exp_a - sim_df.run_exp_b + sim_df.R_a - sim_df.R_b

    lw_df = sim_df.groupby('outcome').run_exp_change.mean()
    lw_df -= lw_df.loc['OUT']
    del lw_df['OUT']

    wOBA_scale = lg_off_df['OBP'] / ((lw_df * lg_off_df.loc[lw_df.index]).sum() / lg_off_df.loc['PA'])
    dump(wOBA_scale, Path('./data/guts/wOBA_scale.joblib'))
    lw_df *= wOBA_scale
    lw_df['wOBA_scale'] = wOBA_scale
    lw_df['runSB'] = 0.2
    lw_df.to_csv('data/guts/linear_weights.csv')
    return lw_df


def determine_rpw():
    pitch_df = load_pitching_data()
    RPW = 9*(pitch_df.R.sum()/pitch_df['IP_dec'].sum())*1.5+3
    dump(RPW, Path('./data/guts/RPW.joblib'))
    return RPW

