import sys
import pandas as pd
from pandas import DataFrame as df, read_csv
import numpy as np
import dtale

SHOW_REG_SEASON_DF = True


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


def show_if():
    if SHOW_REG_SEASON_DF:
        dtale.show(REGULAR_SEASON, subprocess=False)


def forward_fill(column):
    REGULAR_SEASON[column] = REGULAR_SEASON[column].fillna(method='ffill')


def normalize(col, method='minMax'):
    if method == 'minMax':
        REGULAR_SEASON[col] = (REGULAR_SEASON[col] - REGULAR_SEASON[col].min()) / \
            (REGULAR_SEASON[col].max() - REGULAR_SEASON[col].min())
    elif method == 'zScale':
        REGULAR_SEASON[col] = (
            REGULAR_SEASON[col] - REGULAR_SEASON[col].mean() / REGULAR_SEASON[col].std())


def assign_stats(dataframe: df, column: str):
    offensive = column
    defensive = f'd{column}'
    homestat = f'h{column}'
    awaystat = f'a{column}'
    dataframe[offensive] = np.where(
        dataframe['at'] == '@', dataframe[awaystat], dataframe[homestat])
    dataframe[defensive] = np.where(
        dataframe['at'] == '@', dataframe[homestat], dataframe[awaystat])


def parse_home_away_stats(dataframe, spit_col, new_cols):
    data = df(columns=new_cols)
    data[new_cols] = dataframe[spit_col].str.split(
        r'(?<!-)-', expand=True).astype('int')
    return dataframe.join(data)


def rolling_averages(dataframe, column, key, game_span):
    dataframe[column] = dataframe.groupby(['team'])[key].shift(1).rolling(
        game_span).mean().reset_index(0, drop=True)


def get_opp_trail(dataframe, column, key):
    dataframe[column] = [dataframe[key][i] for i in dataframe['opponent_col']]


def determine_home_team(dataframe):
    dataframe['Home_Team'] = np.where(
        dataframe['at'] == '@', dataframe['opponent'], dataframe['team'])
    dataframe['Away_Team'] = np.where(
        dataframe['at'] == '@', dataframe['team'], dataframe['opponent'])
    return dataframe


def determine_home_score(dataframe):
    dataframe['Home_Score'] = np.where(
        dataframe['at'] == '@', dataframe['opp_score'], dataframe['team_score'])
    dataframe['Away_Score'] = np.where(
        dataframe['at'] == '@', dataframe['team_score'], dataframe['opp_score'])
    return dataframe


def parse_vegas_line(dataframe):
    pattern = '^(San |New |St. |Green |Tampa |Kansas |Los |Las )*'
    replace = ''
    dataframe['Vegas_Line_Close'].replace(
        to_replace=pattern, value=replace, inplace=True, regex=True)
    dataframe['Vegas_Line_Close'].replace(
        to_replace='Pick', value='Pick Pick 0', inplace=True, regex=True)
    dataframe['Vegas_Line_Close'].replace(
        to_replace='Football Team', value='Redskins', inplace=True, regex=True)
    dataframe['Vegas_Line_Close'].replace(
        to_replace='Commanders', value="Redskins", inplace=True, regex=True)
    dataframe[['Fav_City', 'Fav_Team', 'Fav_Spread']
              ] = dataframe['Vegas_Line_Close'].str.rsplit(' ', 2, expand=True)
    dataframe['Fav_Spread'] = pd.to_numeric(dataframe['Fav_Spread'])
    return dataframe


def determine_home_spread(dataframe):
    name_dict = {'Jaguars': 'jax', 'Cardinals': 'crd', 'Titans': 'oti',
                'Bears': 'chi', 'Vikings': 'min', 'Seahawks': 'sea', 'Giants': 'nyg', 'Falcons': 'atl', 'Chargers': 'sdg', 'Rams': 'ram', 'Chiefs': 'kan', 'Panthers': 'car', '49ers': 'sfo', 'Cowboys': 'dal', 'Saints': 'nor', 'Broncos': 'den', 'Steelers': 'pit', 'Redskins': 'was', 'Bengals': 'cin', 'Eagles': 'phi', 'Ravens': 'rav', 'Lions': 'det', 'Patriots': 'nwe', 'Packers': 'gnb', 'Jets': 'nyj', 'Buccaneers': 'tam', 'Texans': 'htx', 'Browns': 'cle', 'Pick': '000', 'Dolphins': 'mia', 'Bills': 'buf', 'Raiders': 'rai', 'Colts': 'clt'}

    dataframe['favorite'] = dataframe['Fav_Team'].map(name_dict)

    dataframe['HomeFav'] = (
        dataframe['favorite'] == '000').map({True: 0, False: -1})
    dataframe.loc[dataframe['favorite'] ==
                  dataframe['Home_Team'], 'HomeFav'] = 1
    dataframe['underdog'] = (
        dataframe['favorite'] == dataframe['team']).map({True: 0, False: 1})

    dataframe = dataframe.fillna(0)
    dataframe['Home_Vegas_Spread'] = ([dataframe['Fav_Spread'][i]
                                      if dataframe['HomeFav'][i] == 1 else -1 * dataframe['Fav_Spread'][i] for i in range(len(dataframe))])
    dataframe['Home_Actual_Spread'] = -1 * \
        (dataframe['Home_Score'] - dataframe['Away_Score']
         )
    return dataframe


GAME_SPAN = 10
CURRENT_SEASON_YEAR = 2022
CURRENT_SEASON_WEEK = 2
PATH = '../nfl_master_2009-2022.csv'
NEW_WEEK_PATH = f'../game_docs/games{CURRENT_SEASON_YEAR}-week{CURRENT_SEASON_WEEK}.csv'

RAW_SET: df = pd.read_csv(PATH)
NEW_SET: df = pd.read_csv(NEW_WEEK_PATH)

RAW_SET = RAW_SET.append(NEW_SET, ignore_index=True)


try:
    PLAYOFFS = ['Wild Card', 'Division', 'Conf. Champ.', 'SuperBowl']
    REGULAR_SEASON: df = RAW_SET
    REGULAR_SEASON = REGULAR_SEASON[~REGULAR_SEASON['week'].isin(PLAYOFFS)]
except KeyError as err:
    print(f'unable to drop playoffs columns.\n{err}\ncontinuing...')

weekcount = len(REGULAR_SEASON['week'].unique())
if (weekcount not in [17, 18]):
    print(f'invalid number of weeks: {weekcount}')
    sys.exit()

# TODO: Fix this monstrosity
opponent_names = REGULAR_SEASON['team'].unique(
).tolist()+['ram', 'rai', 'sdg', 'was', 'was']
verbose_names = REGULAR_SEASON['verbose_name'].unique(
).tolist()+['Washington Football Team', 'Washington Commanders']

map = dict(zip(verbose_names, opponent_names))
REGULAR_SEASON['opponent'] = REGULAR_SEASON['opponent'].map(map)
REGULAR_SEASON['at'] = REGULAR_SEASON['at'].fillna('')

opponentcount = len(REGULAR_SEASON['opponent'].unique())

if(opponentcount != 32):
    print(f'invalid number of teams: {opponentcount}')
    # Difference in counts gives good idea of where discrepancy is
    print(REGULAR_SEASON['opponent'].value_counts())
    print(REGULAR_SEASON['opponent'].unique())
    sys.exit()


REGULAR_SEASON = determine_home_team(REGULAR_SEASON)
REGULAR_SEASON = determine_home_score(REGULAR_SEASON)
REGULAR_SEASON = parse_vegas_line(REGULAR_SEASON)
REGULAR_SEASON = determine_home_spread(REGULAR_SEASON)

SPLIT_COLUMNS = {'aCmp-Att-Yd-TD-INT': ['aCmp', 'aAtt', 'aYd', 'aTD', 'aINT'],
                 'hCmp-Att-Yd-TD-INT': ['hCmp', 'hAtt', 'hYd', 'hTD', 'hINT'],
                 'aRush-Yds-Tds': ['aRush', 'aRYds', 'aRTDs'],
                 'hRush-Yds-Tds': ['hRush', 'hRYds', 'hRTDs'],
                 'aPenalties-Yds': ['aPen', 'aPenYds'], 
                 'h-Penalties-Yds': ['hPen', 'hPenYds'],
                 'aThird_Down_Conv': ['aThrdConv', 'aThrd'],
                 'hThird_Down_Conv': ['hThrdConv', 'hThrd'], 
                 'aFourth_Down_Conv': ['aFrthConv', 'aFrth'],
                 'hFourth_Down_Conv': ['hFrthConv', 'hFrth']}


for key, val in SPLIT_COLUMNS.items():
    REGULAR_SEASON = parse_home_away_stats(REGULAR_SEASON, key, val)


passing = ['Cmp', 'Att', 'Yd', 'TD', 'INT']
rushing = ['Rush', 'RYds', 'RTDs']
downs = ['ThrdConv', 'Thrd', 'FrthConv', 'Frth']


for column in passing:
    assign_stats(REGULAR_SEASON, column)

REGULAR_SEASON['team_pass_eff'] = REGULAR_SEASON['Yd'] / REGULAR_SEASON['Att']
REGULAR_SEASON['team_pass_def'] = REGULAR_SEASON['dYd'] / REGULAR_SEASON['dAtt']

for column in rushing:
    assign_stats(REGULAR_SEASON, column)
REGULAR_SEASON['team_rush_eff'] = REGULAR_SEASON['RYds'] / REGULAR_SEASON['Rush']
REGULAR_SEASON['team_rush_def'] = REGULAR_SEASON['dRYds'] / \
    REGULAR_SEASON['dRush']

for column in downs:
    assign_stats(REGULAR_SEASON, column)

for col in [*downs, 'dThrdConv', 'dThrd', 'dFrthConv', 'dFrth']:
    normalize(col, method='zScale')

REGULAR_SEASON['third_eff'] = REGULAR_SEASON['ThrdConv'] / REGULAR_SEASON['Thrd']
REGULAR_SEASON['third_def'] = REGULAR_SEASON['dThrdConv'] / \
    REGULAR_SEASON['dThrd']
REGULAR_SEASON['fourth_eff'] = REGULAR_SEASON['FrthConv'] / REGULAR_SEASON['Frth']
REGULAR_SEASON['fourth_def'] = REGULAR_SEASON['dFrthConv'] / \
    REGULAR_SEASON['dFrth']

fourth_eff_avg = (REGULAR_SEASON[REGULAR_SEASON['fourth_eff'] != float('inf')])[
    'fourth_eff'].mean()
fourth_def_avg = (REGULAR_SEASON[REGULAR_SEASON['fourth_def'] != float('inf')])[
    'fourth_def'].mean()

REGULAR_SEASON['fourth_eff'] = np.where(REGULAR_SEASON['fourth_eff'] == float(
    'inf'), fourth_eff_avg, REGULAR_SEASON['fourth_eff'])
REGULAR_SEASON['fourth_def'] = np.where(REGULAR_SEASON['fourth_def'] == float(
    'inf'), fourth_def_avg, REGULAR_SEASON['fourth_def'])

REGULAR_SEASON['Pen'] = np.where(
    REGULAR_SEASON['at'] == '@', REGULAR_SEASON['aPen'], REGULAR_SEASON['hPen'])
REGULAR_SEASON['PenAgg'] = np.where(
    REGULAR_SEASON['at'] == '@', REGULAR_SEASON['hPen'], REGULAR_SEASON['aPen'])
REGULAR_SEASON['PenYds'] = np.where(
    REGULAR_SEASON['at'] == '@', REGULAR_SEASON['aPenYds'], REGULAR_SEASON['hPenYds'])
REGULAR_SEASON['PenYdsAgg'] = np.where(
    REGULAR_SEASON['at'] == '@', REGULAR_SEASON['hPenYds'], REGULAR_SEASON['aPenYds'])


# create index based on boxscoreUri value - this should allow us to look up weekly matchups
REGULAR_SEASON['opponent_col'] = REGULAR_SEASON['boxscore_url'].groupby(
    REGULAR_SEASON['boxscore_url']).transform(lambda x: np.roll(x.index, 1))


grp = REGULAR_SEASON.groupby(['team'])
# dont know if this works
# todo: try this later
# regularSeason['lostLastAsFav'] = np.where(grp['favorite'].shift(1) == grp['team'].shift(1) & grp['result'] == 'L', 1, 0)
# regularSeason['wonLastAsDog'] = np.where(grp['underdog'].shift(1) == 1 & grp['result'] == 'W', 1, 0)
##


# regularSeason['trail_score'] = grp.tail(game_span).groupby('team')['team_score'].mean()
rolling_averages(REGULAR_SEASON, 'trail_score', 'team_score', GAME_SPAN)
rolling_averages(REGULAR_SEASON, 'trail_allow', 'opp_score', GAME_SPAN)
rolling_averages(REGULAR_SEASON, 'trail_to', 'off_turn_overs', GAME_SPAN)
rolling_averages(REGULAR_SEASON, 'trail_fto', 'def_turn_overs', GAME_SPAN)
rolling_averages(REGULAR_SEASON, 'trail_pass_eff', 'team_pass_eff', GAME_SPAN)
rolling_averages(REGULAR_SEASON, 'trail_pass_def', 'team_pass_def', GAME_SPAN)
rolling_averages(REGULAR_SEASON, 'trail_rush_eff', 'team_rush_eff', GAME_SPAN)
rolling_averages(REGULAR_SEASON, 'trail_rush_def', 'team_rush_def', GAME_SPAN)
rolling_averages(REGULAR_SEASON, 'trail_penYds', 'PenYds', GAME_SPAN)
rolling_averages(REGULAR_SEASON, 'trail_penYdsAgg', 'PenYdsAgg', GAME_SPAN)
rolling_averages(REGULAR_SEASON, 'trail_third_eff', 'third_eff', GAME_SPAN)
rolling_averages(REGULAR_SEASON, 'trail_third_def', 'third_def', GAME_SPAN)
rolling_averages(REGULAR_SEASON, 'trail_fourth_eff', 'fourth_eff', GAME_SPAN)
rolling_averages(REGULAR_SEASON, 'trail_fourth_def', 'fourth_def', GAME_SPAN)


get_opp_trail(REGULAR_SEASON, 'trail_opp_score', 'trail_score')
get_opp_trail(REGULAR_SEASON, 'trail_opp_allow', 'trail_allow')
get_opp_trail(REGULAR_SEASON, 'trail_opp_to', 'trail_to')
get_opp_trail(REGULAR_SEASON, 'trail_opp_fto', 'trail_fto')

get_opp_trail(REGULAR_SEASON, 'trail_opp_pass_eff', 'trail_pass_eff')
get_opp_trail(REGULAR_SEASON, 'trail_opp_pass_def', 'trail_pass_def')
get_opp_trail(REGULAR_SEASON, 'trail_opp_rush_eff', 'trail_rush_eff')
get_opp_trail(REGULAR_SEASON, 'trail_opp_rush_def', 'trail_rush_def')
get_opp_trail(REGULAR_SEASON, 'trail_opp_penYds', 'trail_penYds')
get_opp_trail(REGULAR_SEASON, 'trail_opp_penYdsAgg', 'trail_penYdsAgg')
get_opp_trail(REGULAR_SEASON, 'trail_opp_third_eff', 'trail_third_eff')
get_opp_trail(REGULAR_SEASON, 'trail_opp_third_def', 'trail_third_def')
get_opp_trail(REGULAR_SEASON, 'trail_opp_fourth_eff', 'trail_fourth_eff')
get_opp_trail(REGULAR_SEASON, 'trail_opp_fourth_def', 'trail_fourth_def')


features = df()
features['year'] = REGULAR_SEASON['year']
features['week'] = REGULAR_SEASON['week']
features['Home_Team'] = REGULAR_SEASON['Home_Team']
features['Away_Team'] = REGULAR_SEASON['Away_Team']

features['Home_Win'] = np.NaN
features.loc[features[(REGULAR_SEASON['result'] == 'L') & (
    REGULAR_SEASON['at'] == '')].index, 'Home_Win'] = 'L'
features.loc[features[(REGULAR_SEASON['result'] == 'W') & (
    REGULAR_SEASON['at'] == '')].index, 'Home_Win'] = 'W'
features.loc[features[(REGULAR_SEASON['result'] == 'L') & (
    REGULAR_SEASON['at'] == '@')].index, 'Home_Win'] = 'W'
features.loc[features[(REGULAR_SEASON['result'] == 'W') & (
    REGULAR_SEASON['at'] == '@')].index, 'Home_Win'] = 'L'
# features['Home_Win'] = features.loc[(regularSeason['result'] == 'L') & (regularSeason['at'] == ''), 'Home_Win'] = 'L'
# features['Home_Win'] = features.loc[(regularSeason['result'] == 'W') & (regularSeason['at'] == ''), 'Home_Win'] = 'W'
# features['Home_Win'] = features.loc[(regularSeason['result'] == 'L') & (regularSeason['at'] == '@'), 'Home_Win'] = 'W'
# features['Home_Win'] = features.loc[(regularSeason['result'] == 'W') & (regularSeason['at'] == '@'), 'Home_Win'] = 'L'
# features

features['Home_Fav'] = REGULAR_SEASON['HomeFav']
features['Home_Vegas_Spread'] = REGULAR_SEASON['Home_Vegas_Spread']
features['Home_Actual_Spread'] = REGULAR_SEASON['Home_Actual_Spread']
features['Home_Score'] = REGULAR_SEASON['Home_Score']
features['Away_Score'] = REGULAR_SEASON['Away_Score']
features['Trail_Home_Score'] = np.where(
    REGULAR_SEASON['at'] == '@', REGULAR_SEASON['trail_opp_score'], REGULAR_SEASON['trail_score'])
features['Trail_Away_Score'] = np.where(
    REGULAR_SEASON['at'] == '@', REGULAR_SEASON['trail_score'], REGULAR_SEASON['trail_opp_score'])
features['Home_Allowed'] = np.where(
    REGULAR_SEASON['at'] == '@', REGULAR_SEASON['trail_opp_allow'], REGULAR_SEASON['trail_allow'])
features['Away_Allowed'] = np.where(
    REGULAR_SEASON['at'] == '@', REGULAR_SEASON['trail_allow'], REGULAR_SEASON['trail_opp_allow'])
features['Home_TO'] = np.where(
    REGULAR_SEASON['at'] == '@', REGULAR_SEASON['trail_opp_to'], REGULAR_SEASON['trail_to'])
features['Away_TO'] = np.where(
    REGULAR_SEASON['at'] == '@', REGULAR_SEASON['trail_to'], REGULAR_SEASON['trail_opp_to'])
features['Home_FTO'] = np.where(
    REGULAR_SEASON['at'] == '@', REGULAR_SEASON['trail_opp_fto'], REGULAR_SEASON['trail_fto'])
features['Away_FTO'] = np.where(
    REGULAR_SEASON['at'] == '@', REGULAR_SEASON['trail_fto'], REGULAR_SEASON['trail_opp_fto'])
features['Home_Pass_Eff'] = np.where(
    REGULAR_SEASON['at'] == '@', REGULAR_SEASON['trail_opp_pass_eff'], REGULAR_SEASON['trail_pass_eff'])
features['Away_Pass_Eff'] = np.where(
    REGULAR_SEASON['at'] == '@', REGULAR_SEASON['trail_pass_eff'], REGULAR_SEASON['trail_opp_pass_eff'])
features['Home_Pass_Def'] = np.where(
    REGULAR_SEASON['at'] == '@', REGULAR_SEASON['trail_opp_pass_def'], REGULAR_SEASON['trail_pass_def'])
features['Away_Pass_Def'] = np.where(
    REGULAR_SEASON['at'] == '@', REGULAR_SEASON['trail_pass_def'], REGULAR_SEASON['trail_opp_pass_def'])
features['Home_Rush_Eff'] = np.where(
    REGULAR_SEASON['at'] == '@', REGULAR_SEASON['trail_opp_rush_eff'], REGULAR_SEASON['trail_rush_eff'])
features['Away_Rush_Eff'] = np.where(
    REGULAR_SEASON['at'] == '@', REGULAR_SEASON['trail_rush_eff'], REGULAR_SEASON['trail_opp_rush_eff'])
features['Home_Rush_Def'] = np.where(
    REGULAR_SEASON['at'] == '@', REGULAR_SEASON['trail_opp_rush_def'], REGULAR_SEASON['trail_rush_def'])
features['Away_Rush_Def'] = np.where(
    REGULAR_SEASON['at'] == '@', REGULAR_SEASON['trail_rush_def'], REGULAR_SEASON['trail_opp_rush_def'])
features['Home_Pen_Yds'] = np.where(
    REGULAR_SEASON['at'] == '@', REGULAR_SEASON['trail_opp_penYds'], REGULAR_SEASON['trail_penYds'])
features['Away_Pen_Yds'] = np.where(
    REGULAR_SEASON['at'] == '@', REGULAR_SEASON['trail_penYds'], REGULAR_SEASON['trail_opp_penYds'])
features['Home_Pen_Yds_Agg'] = np.where(
    REGULAR_SEASON['at'] == '@', REGULAR_SEASON['trail_opp_penYdsAgg'], REGULAR_SEASON['trail_penYdsAgg'])
features['Away_Pen_Yds_Agg'] = np.where(
    REGULAR_SEASON['at'] == '@', REGULAR_SEASON['trail_penYdsAgg'], REGULAR_SEASON['trail_opp_penYdsAgg'])
features['Home_Third_Eff'] = np.where(
    REGULAR_SEASON['at'] == '@', REGULAR_SEASON['trail_opp_third_eff'], REGULAR_SEASON['trail_third_eff'])
features['Away_Third_Eff'] = np.where(
    REGULAR_SEASON['at'] == '@', REGULAR_SEASON['trail_third_eff'], REGULAR_SEASON['trail_opp_third_eff'])
features['Home_Third_Def'] = np.where(
    REGULAR_SEASON['at'] == '@', REGULAR_SEASON['trail_opp_third_def'], REGULAR_SEASON['trail_third_def'])
features['Away_Third_Def'] = np.where(
    REGULAR_SEASON['at'] == '@', REGULAR_SEASON['trail_third_def'], REGULAR_SEASON['trail_opp_third_def'])
features['Home_Fourth_Eff'] = np.where(
    REGULAR_SEASON['at'] == '@', REGULAR_SEASON['trail_opp_fourth_eff'], REGULAR_SEASON['trail_fourth_eff'])
features['Away_Fourth_Eff'] = np.where(
    REGULAR_SEASON['at'] == '@', REGULAR_SEASON['trail_fourth_eff'], REGULAR_SEASON['trail_opp_fourth_eff'])
features['Home_Fourth_Def'] = np.where(
    REGULAR_SEASON['at'] == '@', REGULAR_SEASON['trail_opp_fourth_def'], REGULAR_SEASON['trail_fourth_def'])
features['Away_Fourth_Def'] = np.where(
    REGULAR_SEASON['at'] == '@', REGULAR_SEASON['trail_fourth_def'], REGULAR_SEASON['trail_opp_fourth_def'])

features['boxscore_url'] = REGULAR_SEASON['boxscore_url']

features = features.drop_duplicates(subset=['boxscore_url'])

features.to_csv('./cleaned.csv')
