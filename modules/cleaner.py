import sys
import pandas as pd
from pandas import DataFrame as df
import numpy as np
import argparse
from datetime import datetime
from new_week import new_week
import dtale
import config

show_regular_season_df = True
def showIf(data):
    if show_regular_season_df:
        dtale.show(data, subprocess=False)

def print_full(output):
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 2000)
    pd.set_option('display.float_format', '{:20,.2f}'.format)
    pd.set_option('display.max_colwidth', None)
    print(output)
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')
    pd.reset_option('display.width')
    pd.reset_option('display.float_format')
    pd.reset_option('display.max_colwidth')


def forward_fill(df: pd.DataFrame, column):
    df[column] = df[column].fillna(method='ffill')
    return df.copy()


def normalize(df: pd.DataFrame, col, method='minMax'):
    if method == 'minMax':
        df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
    elif method == 'zScale':
        df[col] = df[col] - df[col].mean() / df[col].std()
    return df.copy()


def assign_stats(df: pd.DataFrame, column: str):
    offensive = column
    defensive = f'd{column}'
    homestat = f'h{column}'
    awaystat = f'a{column}'
    df[offensive] = np.where(df['at'] == '@', df[awaystat], df[homestat])
    df[defensive] = np.where(df['at'] == '@', df[homestat], df[awaystat])
    return df.copy()


def parse_home_away_stats(df: pd.DataFrame, spit_col, new_cols, fill_na=0):
    data = pd.DataFrame(columns=new_cols)
    try:
        x = df[spit_col].str.split(r'(?<!-)-', expand=True).astype('int')
        data[new_cols] = x.iloc[:, 0:len(new_cols)]
    except Exception as e:
        print('!! failed to parse stats. manual repair required... ')
        print('... to find the bad columns try importing it to a spreadsheet and filter on Roof or Surface !!')
        
        raise e
    return df.join(data)


def rolling_averages(df: pd.DataFrame, column, key, game_span):
    df[column] = (
        df.groupby(['team'])[key]
        .shift(1)
        .ewm(span=game_span)
        .mean()
        .reset_index(0, drop=True)
    )
    return df.copy()


def get_opp_trail(df, column, key):
    df[column] = [df[key][i] for i in df['opponent_col']]
    return df


def determine_home_team(df):
    df['Home_Team'] = np.where(df['at'] == '@', df['opponent'], df['team'])
    df['Away_Team'] = np.where(df['at'] == '@', df['team'], df['opponent'])
    return df


def determine_home_score(df):
    df['Home_Score'] = np.where(df['at'] == '@', df['opp_score'], df['team_score'])
    df['Away_Score'] = np.where(df['at'] == '@', df['team_score'], df['opp_score'])
    return df


def parse_vegas_line(df):
    pattern = '^(San |New |St. |Green |Tampa |Kansas |Los |Las )*'
    replace = ''
    df['Vegas_Line_Close'].replace(
        to_replace=pattern, value=replace, inplace=True, regex=True
    )
    df['Vegas_Line_Close'].replace(
        to_replace='Pick', value='Pick Pick 0', inplace=True, regex=True
    )
    df['Vegas_Line_Close'].replace(
        to_replace='Football Team', value='Redskins', inplace=True, regex=True
    )
    df['Vegas_Line_Close'].replace(
        to_replace='Commanders', value="Redskins", inplace=True, regex=True
    )
    df[['Fav_City', 'Fav_Team', 'Fav_Spread']] = df['Vegas_Line_Close'].str.rsplit(
        ' ', 2, expand=True
    )
    df['Fav_Spread'] = pd.to_numeric(df['Fav_Spread'])
    return df


def determine_home_spread(df):
    name_dict = {
        'Jaguars': 'jax',
        'Cardinals': 'crd',
        'Titans': 'oti',
        'Bears': 'chi',
        'Vikings': 'min',
        'Seahawks': 'sea',
        'Giants': 'nyg',
        'Falcons': 'atl',
        'Chargers': 'sdg',
        'Rams': 'ram',
        'Chiefs': 'kan',
        'Panthers': 'car',
        '49ers': 'sfo',
        'Cowboys': 'dal',
        'Saints': 'nor',
        'Broncos': 'den',
        'Steelers': 'pit',
        'Redskins': 'was',
        'Bengals': 'cin',
        'Eagles': 'phi',
        'Ravens': 'rav',
        'Lions': 'det',
        'Patriots': 'nwe',
        'Packers': 'gnb',
        'Jets': 'nyj',
        'Buccaneers': 'tam',
        'Texans': 'htx',
        'Browns': 'cle',
        'Pick': '000',
        'Dolphins': 'mia',
        'Bills': 'buf',
        'Raiders': 'rai',
        'Colts': 'clt',
    }

    df['favorite'] = df['Fav_Team'].map(name_dict)

    df['HomeFav'] = (df['favorite'] == '000').map({True: 0, False: -1})
    df.loc[df['favorite'] == df['Home_Team'], 'HomeFav'] = 1
    df['underdog'] = (df['favorite'] == df['team']).map({True: 0, False: 1})

    df = df.fillna(0)
    df['Home_Vegas_Spread'] = [
        df['Fav_Spread'][i] if df['HomeFav'][i] == 1 else -1 * df['Fav_Spread'][i]
        for i in range(len(df))
    ]
    df['Home_Actual_Spread'] = -1 * (df['Home_Score'] - df['Away_Score'])
    return df


def spread_performance(df):
    team_group = df.groupby('team')

    was_favorite = team_group['favorite'].shift(1) == team_group['team'].shift(1)
    was_dog = team_group['underdog'].shift(1) == 1
    lost_last = team_group['result'].shift(1) == 'L'
    won_last = team_group['result'].shift(1) == 'W'
    lost_last_as_favorite = was_favorite & lost_last
    won_last_as_dog = was_dog & won_last
    df['lostLast'] = np.where(lost_last, 1, 0)
    df['wonLast'] = np.where(won_last, 1, 0)
    df['lostLastAsFav'] = np.where(lost_last_as_favorite, 1, 0)
    df['wonLastAsDog'] = np.where(won_last_as_dog, 1, 0)

    return df


def clean(year, week, new_set=pd.DataFrame, game_span=config.GAME_SPAN):  # type: ignore

    current_season_year = year
    current_season_week = week
    print(f'starting cleaner process for {year} {week}')

    raw_set: pd.DataFrame = pd.read_csv(config.MASTER_PATH)

    if new_set.empty:
        new_week_path = (
            f'../game_docs/games{current_season_year}-week{current_season_week}.csv'
        )
        new_set: pd.DataFrame = pd.read_csv(new_week_path)

    raw_set = raw_set.append(new_set, ignore_index=True)
    REGULAR_SEASON: pd.DataFrame = raw_set
    try:
        PLAYOFFS = ['Wild Card', 'Division', 'Conf. Champ.', 'SuperBowl']
        REGULAR_SEASON = REGULAR_SEASON[~REGULAR_SEASON['week'].isin(PLAYOFFS)]
    except KeyError as err:
        print(f'unable to drop playoffs columns.\n{err}\ncontinuing...')

    WEEK_COUNT = len(REGULAR_SEASON['week'].unique())
    if WEEK_COUNT not in [17, 18]:
        print(f'invalid number of weeks: {WEEK_COUNT}')
        sys.exit()

    OPPONENT_NAME_MAP = {k: oldk for oldk, oldv in config.OPPONENT_ABBREV_MAP.items() for k in oldv}

    REGULAR_SEASON['opponent'] = REGULAR_SEASON['opponent'].map(OPPONENT_NAME_MAP)
    REGULAR_SEASON['at'] = REGULAR_SEASON['at'].fillna('')

    OPPONENT_COUNT = len(REGULAR_SEASON['opponent'].unique())

    if OPPONENT_COUNT != 32:
        print(f'invalid number of teams: {OPPONENT_COUNT}')
        # Difference in counts gives good idea of where discrepancy is
        print(REGULAR_SEASON['opponent'].value_counts())
        print(REGULAR_SEASON['opponent'].unique())
        sys.exit()

    REGULAR_SEASON = determine_home_team(REGULAR_SEASON)
    REGULAR_SEASON = determine_home_score(REGULAR_SEASON)
    REGULAR_SEASON = parse_vegas_line(REGULAR_SEASON)
    REGULAR_SEASON = determine_home_spread(REGULAR_SEASON)

    SPLIT_COLUMNS = {
        'aCmp-Att-Yd-TD-INT': ['aCmp', 'aAtt', 'aYd', 'aTD', 'aINT'],
        'hCmp-Att-Yd-TD-INT': ['hCmp', 'hAtt', 'hYd', 'hTD', 'hINT'],
        'aRush-Yds-Tds': ['aRush', 'aRYds', 'aRTDs'],
        'hRush-Yds-Tds': ['hRush', 'hRYds', 'hRTDs'],
        'aPenalties-Yds': ['aPen', 'aPenYds'],
        'h-Penalties-Yds': ['hPen', 'hPenYds'],
        'aThird_Down_Conv': ['aThrdConv', 'aThrd'],
        'hThird_Down_Conv': ['hThrdConv', 'hThrd'],
        'aFourth_Down_Conv': ['aFrthConv', 'aFrth'],
        'hFourth_Down_Conv': ['hFrthConv', 'hFrth'],
    }

    for key, val in SPLIT_COLUMNS.items():
        REGULAR_SEASON = parse_home_away_stats(REGULAR_SEASON, key, val)

    passing = ['Cmp', 'Att', 'Yd', 'TD', 'INT']
    rushing = ['Rush', 'RYds', 'RTDs']
    downs = ['ThrdConv', 'Thrd', 'FrthConv', 'Frth']

    for column in passing:
        REGULAR_SEASON = assign_stats(REGULAR_SEASON, column)

    REGULAR_SEASON['team_pass_eff'] = REGULAR_SEASON['Yd'] / REGULAR_SEASON['Att']
    REGULAR_SEASON['team_pass_def'] = REGULAR_SEASON['dYd'] / REGULAR_SEASON['dAtt']

    for column in rushing:
        REGULAR_SEASON = assign_stats(REGULAR_SEASON, column)
    REGULAR_SEASON['team_rush_eff'] = REGULAR_SEASON['RYds'] / REGULAR_SEASON['Rush']
    REGULAR_SEASON['team_rush_def'] = REGULAR_SEASON['dRYds'] / REGULAR_SEASON['dRush']

    for column in downs:
        REGULAR_SEASON = assign_stats(REGULAR_SEASON, column)

    for col in [*downs, 'dThrdConv', 'dThrd', 'dFrthConv', 'dFrth']:
        REGULAR_SEASON = normalize(REGULAR_SEASON, col, method='zScale')

    REGULAR_SEASON['third_eff'] = REGULAR_SEASON['ThrdConv'] / REGULAR_SEASON['Thrd']
    REGULAR_SEASON['third_def'] = REGULAR_SEASON['dThrdConv'] / REGULAR_SEASON['dThrd']
    REGULAR_SEASON['fourth_eff'] = REGULAR_SEASON['FrthConv'] / REGULAR_SEASON['Frth']
    REGULAR_SEASON['fourth_def'] = REGULAR_SEASON['dFrthConv'] / REGULAR_SEASON['dFrth']

    FOURTH_EFF_AVG = (REGULAR_SEASON[REGULAR_SEASON['fourth_eff'] != float('inf')])[
        'fourth_eff'
    ].mean()
    FOURTH_DEF_AVG = (REGULAR_SEASON[REGULAR_SEASON['fourth_def'] != float('inf')])[
        'fourth_def'
    ].mean()

    REGULAR_SEASON['fourth_eff'] = np.where(
        REGULAR_SEASON['fourth_eff'] == float('inf'),
        FOURTH_EFF_AVG,
        REGULAR_SEASON['fourth_eff'],
    )
    REGULAR_SEASON['fourth_def'] = np.where(
        REGULAR_SEASON['fourth_def'] == float('inf'),
        FOURTH_DEF_AVG,
        REGULAR_SEASON['fourth_def'],
    )

    REGULAR_SEASON['Pen'] = np.where(
        REGULAR_SEASON['at'] == '@', REGULAR_SEASON['aPen'], REGULAR_SEASON['hPen']
    )
    REGULAR_SEASON['PenAgg'] = np.where(
        REGULAR_SEASON['at'] == '@', REGULAR_SEASON['hPen'], REGULAR_SEASON['aPen']
    )
    REGULAR_SEASON['PenYds'] = np.where(
        REGULAR_SEASON['at'] == '@',
        REGULAR_SEASON['aPenYds'],
        REGULAR_SEASON['hPenYds'],
    )
    REGULAR_SEASON['PenYdsAgg'] = np.where(
        REGULAR_SEASON['at'] == '@',
        REGULAR_SEASON['hPenYds'],
        REGULAR_SEASON['aPenYds'],
    )

    # create index based on boxscoreUri value - this should allow us to look up weekly matchups
    REGULAR_SEASON['opponent_col'] = (
        REGULAR_SEASON['boxscore_url']
        .groupby(REGULAR_SEASON['boxscore_url'])
        .transform(lambda x: np.roll(x.index, 1))
    )

    REGULAR_SEASON = spread_performance(REGULAR_SEASON)
    REGULAR_SEASON = rolling_averages(
        REGULAR_SEASON, 'trail_score', 'team_score', game_span
    )
    REGULAR_SEASON = rolling_averages(
        REGULAR_SEASON, 'trail_allow', 'opp_score', game_span
    )
    REGULAR_SEASON = rolling_averages(
        REGULAR_SEASON, 'trail_to', 'off_turn_overs', game_span
    )
    REGULAR_SEASON = rolling_averages(
        REGULAR_SEASON, 'trail_fto', 'def_turn_overs', game_span
    )
    REGULAR_SEASON = rolling_averages(
        REGULAR_SEASON, 'trail_pass_eff', 'team_pass_eff', game_span
    )
    REGULAR_SEASON = rolling_averages(
        REGULAR_SEASON, 'trail_pass_def', 'team_pass_def', game_span
    )
    REGULAR_SEASON = rolling_averages(
        REGULAR_SEASON, 'trail_rush_eff', 'team_rush_eff', game_span
    )
    REGULAR_SEASON = rolling_averages(
        REGULAR_SEASON, 'trail_rush_def', 'team_rush_def', game_span
    )
    REGULAR_SEASON = rolling_averages(
        REGULAR_SEASON, 'trail_penYds', 'PenYds', game_span
    )
    REGULAR_SEASON = rolling_averages(
        REGULAR_SEASON, 'trail_penYdsAgg', 'PenYdsAgg', game_span
    )
    REGULAR_SEASON = rolling_averages(
        REGULAR_SEASON, 'trail_third_eff', 'third_eff', game_span
    )
    REGULAR_SEASON = rolling_averages(
        REGULAR_SEASON, 'trail_third_def', 'third_def', game_span
    )
    REGULAR_SEASON = rolling_averages(
        REGULAR_SEASON, 'trail_fourth_eff', 'fourth_eff', game_span
    )
    REGULAR_SEASON = rolling_averages(
        REGULAR_SEASON, 'trail_fourth_def', 'fourth_def', game_span
    )

    REGULAR_SEASON = get_opp_trail(REGULAR_SEASON, 'trail_opp_score', 'trail_score')
    REGULAR_SEASON = get_opp_trail(REGULAR_SEASON, 'trail_opp_allow', 'trail_allow')
    REGULAR_SEASON = get_opp_trail(REGULAR_SEASON, 'trail_opp_to', 'trail_to')
    REGULAR_SEASON = get_opp_trail(REGULAR_SEASON, 'trail_opp_fto', 'trail_fto')

    REGULAR_SEASON = get_opp_trail(
        REGULAR_SEASON, 'trail_opp_pass_eff', 'trail_pass_eff'
    )
    REGULAR_SEASON = get_opp_trail(
        REGULAR_SEASON, 'trail_opp_pass_def', 'trail_pass_def'
    )
    REGULAR_SEASON = get_opp_trail(
        REGULAR_SEASON, 'trail_opp_rush_eff', 'trail_rush_eff'
    )
    REGULAR_SEASON = get_opp_trail(
        REGULAR_SEASON, 'trail_opp_rush_def', 'trail_rush_def'
    )
    REGULAR_SEASON = get_opp_trail(REGULAR_SEASON, 'trail_opp_penYds', 'trail_penYds')
    REGULAR_SEASON = get_opp_trail(
        REGULAR_SEASON, 'trail_opp_penYdsAgg', 'trail_penYdsAgg'
    )
    REGULAR_SEASON = get_opp_trail(
        REGULAR_SEASON, 'trail_opp_third_eff', 'trail_third_eff'
    )
    REGULAR_SEASON = get_opp_trail(
        REGULAR_SEASON, 'trail_opp_third_def', 'trail_third_def'
    )
    REGULAR_SEASON = get_opp_trail(
        REGULAR_SEASON, 'trail_opp_fourth_eff', 'trail_fourth_eff'
    )
    REGULAR_SEASON = get_opp_trail(
        REGULAR_SEASON, 'trail_opp_fourth_def', 'trail_fourth_def'
    )

    FEATURES = pd.DataFrame()
    FEATURES['year'] = REGULAR_SEASON['year']
    FEATURES['week'] = REGULAR_SEASON['week']
    FEATURES['Home_Team'] = REGULAR_SEASON['Home_Team']
    FEATURES['Away_Team'] = REGULAR_SEASON['Away_Team']

    FEATURES['Home_Win'] = np.NaN
    FEATURES.loc[
        FEATURES[
            (REGULAR_SEASON['result'] == 'L') & (REGULAR_SEASON['at'] == '')
        ].index,
        'Home_Win',
    ] = 'L'
    FEATURES.loc[
        FEATURES[
            (REGULAR_SEASON['result'] == 'W') & (REGULAR_SEASON['at'] == '')
        ].index,
        'Home_Win',
    ] = 'W'
    FEATURES.loc[
        FEATURES[
            (REGULAR_SEASON['result'] == 'L') & (REGULAR_SEASON['at'] == '@')
        ].index,
        'Home_Win',
    ] = 'W'
    FEATURES.loc[
        FEATURES[
            (REGULAR_SEASON['result'] == 'W') & (REGULAR_SEASON['at'] == '@')
        ].index,
        'Home_Win',
    ] = 'L'

    FEATURES['Home_Fav'] = REGULAR_SEASON['HomeFav']
    FEATURES['Home_Vegas_Spread'] = REGULAR_SEASON['Home_Vegas_Spread']
    FEATURES['Home_Actual_Spread'] = REGULAR_SEASON['Home_Actual_Spread']
    FEATURES['Home_Score'] = REGULAR_SEASON['Home_Score']
    FEATURES['Away_Score'] = REGULAR_SEASON['Away_Score']
    FEATURES['Trail_Home_Score'] = np.where(
        REGULAR_SEASON['at'] == '@',
        REGULAR_SEASON['trail_opp_score'],
        REGULAR_SEASON['trail_score'],
    )
    FEATURES['Trail_Away_Score'] = np.where(
        REGULAR_SEASON['at'] == '@',
        REGULAR_SEASON['trail_score'],
        REGULAR_SEASON['trail_opp_score'],
    )
    FEATURES['Home_Allowed'] = np.where(
        REGULAR_SEASON['at'] == '@',
        REGULAR_SEASON['trail_opp_allow'],
        REGULAR_SEASON['trail_allow'],
    )
    FEATURES['Away_Allowed'] = np.where(
        REGULAR_SEASON['at'] == '@',
        REGULAR_SEASON['trail_allow'],
        REGULAR_SEASON['trail_opp_allow'],
    )
    FEATURES['Home_TO'] = np.where(
        REGULAR_SEASON['at'] == '@',
        REGULAR_SEASON['trail_opp_to'],
        REGULAR_SEASON['trail_to'],
    )
    FEATURES['Away_TO'] = np.where(
        REGULAR_SEASON['at'] == '@',
        REGULAR_SEASON['trail_to'],
        REGULAR_SEASON['trail_opp_to'],
    )
    FEATURES['Home_FTO'] = np.where(
        REGULAR_SEASON['at'] == '@',
        REGULAR_SEASON['trail_opp_fto'],
        REGULAR_SEASON['trail_fto'],
    )
    FEATURES['Away_FTO'] = np.where(
        REGULAR_SEASON['at'] == '@',
        REGULAR_SEASON['trail_fto'],
        REGULAR_SEASON['trail_opp_fto'],
    )
    FEATURES['Home_Pass_Eff'] = np.where(
        REGULAR_SEASON['at'] == '@',
        REGULAR_SEASON['trail_opp_pass_eff'],
        REGULAR_SEASON['trail_pass_eff'],
    )
    FEATURES['Away_Pass_Eff'] = np.where(
        REGULAR_SEASON['at'] == '@',
        REGULAR_SEASON['trail_pass_eff'],
        REGULAR_SEASON['trail_opp_pass_eff'],
    )
    FEATURES['Home_Pass_Def'] = np.where(
        REGULAR_SEASON['at'] == '@',
        REGULAR_SEASON['trail_opp_pass_def'],
        REGULAR_SEASON['trail_pass_def'],
    )
    FEATURES['Away_Pass_Def'] = np.where(
        REGULAR_SEASON['at'] == '@',
        REGULAR_SEASON['trail_pass_def'],
        REGULAR_SEASON['trail_opp_pass_def'],
    )
    FEATURES['Home_Rush_Eff'] = np.where(
        REGULAR_SEASON['at'] == '@',
        REGULAR_SEASON['trail_opp_rush_eff'],
        REGULAR_SEASON['trail_rush_eff'],
    )
    FEATURES['Away_Rush_Eff'] = np.where(
        REGULAR_SEASON['at'] == '@',
        REGULAR_SEASON['trail_rush_eff'],
        REGULAR_SEASON['trail_opp_rush_eff'],
    )
    FEATURES['Home_Rush_Def'] = np.where(
        REGULAR_SEASON['at'] == '@',
        REGULAR_SEASON['trail_opp_rush_def'],
        REGULAR_SEASON['trail_rush_def'],
    )
    FEATURES['Away_Rush_Def'] = np.where(
        REGULAR_SEASON['at'] == '@',
        REGULAR_SEASON['trail_rush_def'],
        REGULAR_SEASON['trail_opp_rush_def'],
    )
    FEATURES['Home_Pen_Yds'] = np.where(
        REGULAR_SEASON['at'] == '@',
        REGULAR_SEASON['trail_opp_penYds'],
        REGULAR_SEASON['trail_penYds'],
    )
    FEATURES['Away_Pen_Yds'] = np.where(
        REGULAR_SEASON['at'] == '@',
        REGULAR_SEASON['trail_penYds'],
        REGULAR_SEASON['trail_opp_penYds'],
    )
    FEATURES['Home_Pen_Yds_Agg'] = np.where(
        REGULAR_SEASON['at'] == '@',
        REGULAR_SEASON['trail_opp_penYdsAgg'],
        REGULAR_SEASON['trail_penYdsAgg'],
    )
    FEATURES['Away_Pen_Yds_Agg'] = np.where(
        REGULAR_SEASON['at'] == '@',
        REGULAR_SEASON['trail_penYdsAgg'],
        REGULAR_SEASON['trail_opp_penYdsAgg'],
    )
    FEATURES['Home_Third_Eff'] = np.where(
        REGULAR_SEASON['at'] == '@',
        REGULAR_SEASON['trail_opp_third_eff'],
        REGULAR_SEASON['trail_third_eff'],
    )
    FEATURES['Away_Third_Eff'] = np.where(
        REGULAR_SEASON['at'] == '@',
        REGULAR_SEASON['trail_third_eff'],
        REGULAR_SEASON['trail_opp_third_eff'],
    )
    FEATURES['Home_Third_Def'] = np.where(
        REGULAR_SEASON['at'] == '@',
        REGULAR_SEASON['trail_opp_third_def'],
        REGULAR_SEASON['trail_third_def'],
    )
    FEATURES['Away_Third_Def'] = np.where(
        REGULAR_SEASON['at'] == '@',
        REGULAR_SEASON['trail_third_def'],
        REGULAR_SEASON['trail_opp_third_def'],
    )
    FEATURES['Home_Fourth_Eff'] = np.where(
        REGULAR_SEASON['at'] == '@',
        REGULAR_SEASON['trail_opp_fourth_eff'],
        REGULAR_SEASON['trail_fourth_eff'],
    )
    FEATURES['Away_Fourth_Eff'] = np.where(
        REGULAR_SEASON['at'] == '@',
        REGULAR_SEASON['trail_fourth_eff'],
        REGULAR_SEASON['trail_opp_fourth_eff'],
    )
    FEATURES['Home_Fourth_Def'] = np.where(
        REGULAR_SEASON['at'] == '@',
        REGULAR_SEASON['trail_opp_fourth_def'],
        REGULAR_SEASON['trail_fourth_def'],
    )
    FEATURES['Away_Fourth_Def'] = np.where(
        REGULAR_SEASON['at'] == '@',
        REGULAR_SEASON['trail_fourth_def'],
        REGULAR_SEASON['trail_opp_fourth_def'],
    )

    FEATURES['boxscore_url'] = REGULAR_SEASON['boxscore_url']
    FEATURES['wonLast'] = REGULAR_SEASON['wonLast']
    FEATURES['lostLast'] = REGULAR_SEASON['lostLast']
    FEATURES['lostLastAsFav'] = REGULAR_SEASON['lostLastAsFav']
    FEATURES['wonLastAsDog'] = REGULAR_SEASON['wonLastAsDog']

    FEATURES = FEATURES.drop_duplicates(subset=['boxscore_url'])

    FEATURES.to_csv('./cleaned.csv')


if __name__ == '__main__':
    parsed = argparse.ArgumentParser()
    parsed.add_argument('-y', '--year', type=int, default=datetime.now().year)
    parsed.add_argument('-w', '--week', type=int, default=1)
    args = parsed.parse_args()
    week, year = args.week, args.year
    df = new_week(year, week)
    clean(year, week)
