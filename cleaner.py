from cmath import nan
import csv
import pandas as pd
from pandas import DataFrame as df, read_csv
import sys
import numpy as np
import re


def print_full(x):
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 2000)
    pd.set_option('display.float_format', '{:20,.2f}'.format)
    pd.set_option('display.max_colwidth', None)
    print(x)
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')
    pd.reset_option('display.width')
    pd.reset_option('display.float_format')
    pd.reset_option('display.max_colwidth')


game_span = 10
stopifnot2009 = 2009
currentSeasonYear = 2019
path = './nfl_master_2009-2021.csv'
currentSeason = None  # './nfl_master_2019-2019.csv'
newWeekPath = './nfl_newWeek.csv'

rawSet: df = pd.read_csv(path)
newSet: df = pd.read_csv(
    currentSeason) if currentSeason else pd.read_csv(newWeekPath)

# newSet['week'] should be treated as a string

rawSet.append(newSet)

try:
    playoffs = ['Wild Card', 'Division', 'Conf. Champ.', 'SuperBowl']
    regularSeason: df = rawSet
    regularSeason = regularSeason[~regularSeason['week'].isin(playoffs)]
except KeyError as e:
    print(f'unable to drop playoffs columns.\n{e}\ncontinuing...')

weekcount = len(regularSeason['week'].unique())
if (weekcount not in [17,18]):
    print(f'invalid number of weeks: {weekcount}')
    sys.exit()

opponent_names = regularSeason['team'].unique(
).tolist()+['sdg', 'ram', 'rai', 'was']
verbose_names = regularSeason['verbose_name'].unique(
).tolist()+['Los Angeles Chargers', 'Los Angeles Rams', 'Las Vegas Raiders', 'Washington Football Team']

map = dict(zip(verbose_names, opponent_names))
regularSeason['opponent'] = regularSeason['opponent'].map(map)

opponentcount = len(regularSeason['opponent'].unique())

if(opponentcount != 32):
    print(f'invalid number of teams: {opponentcount}')
    # Difference in counts gives good idea of where discrepancy is
    print(regularSeason['opponent'].value_counts())
    sys.exit()

# currentYear = regularSeason[regularSeason['year'] == currentSeasonYear]
# currentYear.to_csv('nfl_newWeek.csv', index=False)

# try:
#     newWeek: df = pd.read_csv(newWeekPath)
#     regularSeason = regularSeason[regularSeason['year'] < currentSeasonYear].append(newWeek)
# except:
#     print('new week file not found!')
#regularSeason.loc[regularSeason['at'] == '@', 'Home_Team'] = regularSeason['opponent']
#regularSeason.loc[regularSeason['at'] == '@', 'Away_Team'] = regularSeason['team']
regularSeason['Home_Team'] = np.where(
    regularSeason['at'] == '@', regularSeason['opponent'], regularSeason['team'])
regularSeason['Away_Team'] = np.where(
    regularSeason['at'] == '@', regularSeason['team'], regularSeason['opponent'])

regularSeason['Home_Score'] = np.where(
    regularSeason['at'] == '@', regularSeason['opp_score'], regularSeason['team_score'])
regularSeason['Away_Score'] = np.where(
    regularSeason['at'] == '@', regularSeason['team_score'], regularSeason['opp_score'])

pattern = '^(San |New |St. |Green |Tampa |Kansas |Los |Las )*'
replace = ''
regularSeason['Vegas_Line_Close'].replace(
    to_replace=pattern, value=replace, inplace=True, regex=True)
regularSeason['Vegas_Line_Close'].replace(
    to_replace='Pick', value='Pick Pick 0', inplace=True, regex=True)
regularSeason['Vegas_Line_Close'].replace(
    to_replace='Football Team', value='Redskins', inplace=True, regex=True)

split = regularSeason['Vegas_Line_Close'].str.rsplit(' ', 2, expand=True)

regularSeason[['Fav_City', 'Fav_Team', 'Fav_Spread']
              ] = split
#regularSeason['Fav_Spread'] = regularSeason['Vegas_Line_Close'].str.split(pat=' ', expand=True)[2]

v = regularSeason.loc[1, 'year']
if(v != stopifnot2009):
    print(f'stopping! {v} != {stopifnot2009}')
    sys.exit()

nameDict = {'Jaguars': 'jax', 'Cardinals': 'crd', 'Titans': 'oti',
            'Bears': 'chi', 'Vikings': 'min', 'Seahawks': 'sea', 'Giants': 'nyg', 'Falcons': 'atl', 'Chargers': 'sdg', 'Rams': 'ram', 'Chiefs': 'kan', 'Panthers': 'car', '49ers': 'sfo', 'Cowboys': 'dal', 'Saints': 'nor', 'Broncos': 'den', 'Steelers': 'pit', 'Redskins': 'was', 'Bengals': 'cin', 'Eagles': 'phi', 'Ravens': 'rav', 'Lions': 'det', 'Patriots': 'nwe', 'Packers': 'gnb', 'Jets': 'nyj', 'Buccaneers': 'tam', 'Texans': 'htx', 'Browns': 'cle', 'Pick': '000', 'Dolphins': 'mia', 'Bills': 'buf', 'Raiders': 'rai', 'Colts': 'clt'}

regularSeason['favorite'] = regularSeason['Fav_Team'].map(nameDict)

regularSeason['HomeFav'] = (
    regularSeason['favorite'] == '000').map({True: 0, False: -1})
regularSeason.loc[regularSeason['favorite'] ==
                  regularSeason['Home_Team'], 'HomeFav'] = 1
regularSeason['underdog'] = (
    regularSeason['favorite'] == regularSeason['team']).map({True: 0, False: 1})

regularSeason.fillna(0)

regularSeason['Home_Vegas_Spread'] = regularSeason[regularSeason['HomeFav']
                                                   == 1]['Fav_Spread']
regularSeason['Home_Vegas_Spread'] = np.where(
    regularSeason['HomeFav'] == 0, 0, regularSeason['Fav_Spread'].transform(lambda fav: fav * -1))
regularSeason['Home_Actual_Spread'] = -1 * \
    (regularSeason['Home_Score'] - regularSeason['Away_Score'])


def parse_home_away_stats(dataframe, cols, parse):
    data = df(columns=cols)
    data[cols] = dataframe[parse].str.split(
        r'(?<!-)-', expand=True).astype('int')
    return dataframe.join(data)


regularSeason = parse_home_away_stats(
    regularSeason, ['aCmp', 'aAtt', 'aYd', 'aTD', 'aINT'], 'aCmp-Att-Yd-TD-INT')
regularSeason = parse_home_away_stats(
    regularSeason, ['hCmp', 'hAtt', 'hYd', 'hTD', 'hINT'], 'hCmp-Att-Yd-TD-INT')

regularSeason = parse_home_away_stats(
    regularSeason, ['aRush', 'aRYds', 'aRTDs'], 'aRush-Yds-Tds')
regularSeason = parse_home_away_stats(
    regularSeason, ['hRush', 'hRYds', 'hRTDs'], 'hRush-Yds-Tds')

regularSeason = parse_home_away_stats(
    regularSeason, ['aPen', 'aPenYds'], 'aPenalties-Yds')
regularSeason = parse_home_away_stats(
    regularSeason, ['hPen', 'hPenYds'], 'h-Penalties-Yds')

regularSeason = parse_home_away_stats(
    regularSeason, ['aThrdConv', 'aThrd'], 'aThird_Down_Conv')
regularSeason = parse_home_away_stats(
    regularSeason, ['hThrdConv', 'hThrd'], 'hThird_Down_Conv')

regularSeason = parse_home_away_stats(
    regularSeason, ['aFrthConv', 'aFrth'], 'aFourth_Down_Conv')
regularSeason = parse_home_away_stats(
    regularSeason, ['hFrthConv', 'hFrth'], 'hFourth_Down_Conv')


def assignStats(regularSeason: df, column: str):
    offensive = column
    defensive = f'd{column}'
    homestat = f'h{column}'
    awaystat = f'a{column}'
    regularSeason[offensive] = np.where(
        regularSeason['at'] == '@', regularSeason[awaystat], regularSeason[homestat])
    regularSeason[defensive] = np.where(
        regularSeason['at'] == '@', regularSeason[homestat], regularSeason[awaystat])


passing = ['Cmp', 'Att', 'Yd', 'TD', 'INT']
rushing = ['Rush', 'RYds', 'RTDs']
downs = ['ThrdConv', 'Thrd', 'FrthConv', 'Frth']

for column in passing:
    assignStats(regularSeason, column)

regularSeason['team_pass_eff'] = regularSeason['Yd'] / regularSeason['Att']
regularSeason['team_pass_def'] = regularSeason['dYd'] / regularSeason['dAtt']

for column in rushing:
    assignStats(regularSeason, column)
regularSeason['team_rush_eff'] = regularSeason['RYds'] / regularSeason['Rush']
regularSeason['team_rush_def'] = regularSeason['dRYds'] / \
    regularSeason['dRush']

for column in downs:
    assignStats(regularSeason, column)
regularSeason['third_eff'] = regularSeason['ThrdConv'] / regularSeason['Thrd']
regularSeason['third_def'] = regularSeason['dThrdConv'] / \
    regularSeason['dThrd']
regularSeason['fourth_eff'] = regularSeason['FrthConv'] / regularSeason['Frth']
regularSeason['fourth_def'] = regularSeason['dFrthConv'] / \
    regularSeason['dFrth']

# TODO: figure out a better way to handle divide by zero
regularSeason['fourth_eff'] = np.where(regularSeason['fourth_eff'] == float(
    'inf'), 0.0001, regularSeason['fourth_eff'])
regularSeason['fourth_def'] = np.where(regularSeason['fourth_def'] == float(
    'inf'), 0.0001, regularSeason['fourth_def'])

regularSeason['Pen'] = np.where(
    regularSeason['at'] == '@', regularSeason['aPen'], regularSeason['hPen'])
regularSeason['PenAgg'] = np.where(
    regularSeason['at'] == '@', regularSeason['hPen'], regularSeason['aPen'])
regularSeason['PenYds'] = np.where(
    regularSeason['at'] == '@', regularSeason['aPenYds'], regularSeason['hPenYds'])
regularSeason['PenYdsAgg'] = np.where(
    regularSeason['at'] == '@', regularSeason['hPenYds'], regularSeason['aPenYds'])

# create index based on boxscoreUri value - this should allow us to look up weekly matchups
print(list(regularSeason.keys()))
def findOpponentCol(row, df: pd.DataFrame):
    boxscore_url = row['boxscore_url']
    this_team = row['team']
    matches: pd.Series = ((df['boxscore_url'] == boxscore_url) & (df['team'] != this_team))
    i = df[matches].index
    if len(i) == 0:
        print(f'could not find {boxscore_url}')
        return None
    return i[0]
regularSeason['opponent_col'] = regularSeason[['team','boxscore_url']].apply(lambda row: findOpponentCol(row, regularSeason[['team', 'boxscore_url']]), axis=1)

grp = regularSeason.groupby(['team'])
# dont know if this works
# todo: try this later
#regularSeason['lostLastAsFav'] = np.where(grp['favorite'].shift(1) == grp['team'].shift(1) & grp['result'] == 'L', 1, 0)
#regularSeason['wonLastAsDog'] = np.where(grp['underdog'].shift(1) == 1 & grp['result'] == 'W', 1, 0)
##


def rolling_averages(dataframe, column, key, game_span):
    dataframe[column] = dataframe.groupby(['team'])[key].rolling(
        game_span).mean().reset_index(0, drop=True)


#regularSeason['trail_score'] = grp.tail(game_span).groupby('team')['team_score'].mean()
rolling_averages(regularSeason, 'trail_score', 'team_score', game_span)
rolling_averages(regularSeason, 'trail_allow', 'opp_score', game_span)
rolling_averages(regularSeason, 'trail_to', 'off_turn_overs', game_span)
rolling_averages(regularSeason, 'trail_fto', 'def_turn_overs', game_span)
rolling_averages(regularSeason, 'trail_pass_eff', 'team_pass_eff', game_span)
rolling_averages(regularSeason, 'trail_pass_def', 'team_pass_def', game_span)
rolling_averages(regularSeason, 'trail_rush_eff', 'team_rush_eff', game_span)
rolling_averages(regularSeason, 'trail_rush_def', 'team_rush_def', game_span)
rolling_averages(regularSeason, 'trail_penYds', 'PenYds', game_span)
rolling_averages(regularSeason, 'trail_penYdsAgg', 'PenYdsAgg', game_span)
rolling_averages(regularSeason, 'trail_third_eff', 'third_eff', game_span)
rolling_averages(regularSeason, 'trail_third_def', 'third_def', game_span)
rolling_averages(regularSeason, 'trail_fourth_eff', 'fourth_eff', game_span)
rolling_averages(regularSeason, 'trail_fourth_def', 'fourth_def', game_span)


def get_opp_trail(dataframe, column, key):
    dataframe[column] = [dataframe[key][i] for i in dataframe['opponent_col']]


get_opp_trail(regularSeason, 'trail_opp_score', 'trail_score')
get_opp_trail(regularSeason, 'trail_opp_allow', 'trail_allow')
get_opp_trail(regularSeason, 'trail_opp_to', 'trail_to')
get_opp_trail(regularSeason, 'trail_opp_fto', 'trail_fto')
get_opp_trail(regularSeason, 'trail_opp_pass_eff', 'trail_pass_eff')
get_opp_trail(regularSeason, 'trail_opp_pass_def', 'trail_pass_def')
get_opp_trail(regularSeason, 'trail_opp_rush_eff', 'trail_rush_eff')
get_opp_trail(regularSeason, 'trail_opp_rush_def', 'trail_rush_def')
get_opp_trail(regularSeason, 'trail_opp_penYds', 'trail_penYds')
get_opp_trail(regularSeason, 'trail_opp_penYdsAgg', 'trail_penYdsAgg')
get_opp_trail(regularSeason, 'trail_opp_third_eff', 'trail_third_eff')
get_opp_trail(regularSeason, 'trail_opp_third_def', 'trail_third_def')
get_opp_trail(regularSeason, 'trail_opp_fourth_eff', 'trail_fourth_eff')
get_opp_trail(regularSeason, 'trail_opp_fourth_def', 'trail_fourth_def')

features = df()
features['year'] = regularSeason['year']
features['week'] = regularSeason['week']
features['Home_Team'] = regularSeason['Home_Team']
features['Away_Team'] = regularSeason['Away_Team']

features['Home_Win'] = np.NaN
features.loc[features[(regularSeason['result'] == 'L') & (
    regularSeason['at'] == '')].index, 'Home_Win'] = 'L'
features.loc[features[(regularSeason['result'] == 'W') & (
    regularSeason['at'] == '')].index, 'Home_Win'] = 'W'
features.loc[features[(regularSeason['result'] == 'L') & (
    regularSeason['at'] == '@')].index, 'Home_Win'] = 'W'
features.loc[features[(regularSeason['result'] == 'W') & (
    regularSeason['at'] == '@')].index, 'Home_Win'] = 'L'
#features['Home_Win'] = features.loc[(regularSeason['result'] == 'L') & (regularSeason['at'] == ''), 'Home_Win'] = 'L'
#features['Home_Win'] = features.loc[(regularSeason['result'] == 'W') & (regularSeason['at'] == ''), 'Home_Win'] = 'W'
#features['Home_Win'] = features.loc[(regularSeason['result'] == 'L') & (regularSeason['at'] == '@'), 'Home_Win'] = 'W'
#features['Home_Win'] = features.loc[(regularSeason['result'] == 'W') & (regularSeason['at'] == '@'), 'Home_Win'] = 'L'
# features

features['Home_Fav'] = regularSeason['HomeFav']
features['Home_Vegas_Spread'] = regularSeason['Home_Vegas_Spread']
features['Home_Actual_Spread'] = regularSeason['Home_Actual_Spread']
features['Home_Score'] = regularSeason['Home_Score']
features['Away_Score'] = regularSeason['Away_Score']
features['Trail_Home_Score'] = np.where(
    regularSeason['at'] == '@', regularSeason['trail_opp_score'], regularSeason['trail_score'])
features['Trail_Away_Score'] = np.where(
    regularSeason['at'] == '@', regularSeason['trail_score'], regularSeason['trail_opp_score'])
features['Home_Allowed'] = np.where(
    regularSeason['at'] == '@', regularSeason['trail_opp_allow'], regularSeason['trail_allow'])
features['Away_Allowed'] = np.where(
    regularSeason['at'] == '@', regularSeason['trail_allow'], regularSeason['trail_opp_allow'])
features['Home_TO'] = np.where(
    regularSeason['at'] == '@', regularSeason['trail_opp_to'], regularSeason['trail_to'])
features['Away_TO'] = np.where(
    regularSeason['at'] == '@', regularSeason['trail_to'], regularSeason['trail_opp_to'])
features['Home_FTO'] = np.where(
    regularSeason['at'] == '@', regularSeason['trail_opp_fto'], regularSeason['trail_fto'])
features['Away_FTO'] = np.where(
    regularSeason['at'] == '@', regularSeason['trail_fto'], regularSeason['trail_opp_fto'])
features['Home_Pass_Eff'] = np.where(
    regularSeason['at'] == '@', regularSeason['trail_opp_pass_eff'], regularSeason['trail_pass_eff'])
features['Away_Pass_Eff'] = np.where(
    regularSeason['at'] == '@', regularSeason['trail_pass_eff'], regularSeason['trail_opp_pass_eff'])
features['Home_Pass_Def'] = np.where(
    regularSeason['at'] == '@', regularSeason['trail_opp_pass_def'], regularSeason['trail_pass_def'])
features['Away_Pass_Def'] = np.where(
    regularSeason['at'] == '@', regularSeason['trail_pass_def'], regularSeason['trail_opp_pass_def'])
features['Home_Rush_Eff'] = np.where(
    regularSeason['at'] == '@', regularSeason['trail_opp_rush_eff'], regularSeason['trail_rush_eff'])
features['Away_Rush_Eff'] = np.where(
    regularSeason['at'] == '@', regularSeason['trail_rush_eff'], regularSeason['trail_opp_rush_eff'])
features['Home_Rush_Def'] = np.where(
    regularSeason['at'] == '@', regularSeason['trail_opp_rush_def'], regularSeason['trail_rush_def'])
features['Away_Rush_Def'] = np.where(
    regularSeason['at'] == '@', regularSeason['trail_rush_def'], regularSeason['trail_opp_rush_def'])
features['Home_Pen_Yds'] = np.where(
    regularSeason['at'] == '@', regularSeason['trail_opp_penYds'], regularSeason['trail_penYds'])
features['Away_Pen_Yds'] = np.where(
    regularSeason['at'] == '@', regularSeason['trail_penYds'], regularSeason['trail_opp_penYds'])
features['Home_Pen_Yds_Agg'] = np.where(
    regularSeason['at'] == '@', regularSeason['trail_opp_penYdsAgg'], regularSeason['trail_penYdsAgg'])
features['Away_Pen_Yds_Agg'] = np.where(
    regularSeason['at'] == '@', regularSeason['trail_penYdsAgg'], regularSeason['trail_opp_penYdsAgg'])
features['Home_Third_Eff'] = np.where(
    regularSeason['at'] == '@', regularSeason['trail_opp_third_eff'], regularSeason['trail_third_eff'])
features['Away_Third_Eff'] = np.where(
    regularSeason['at'] == '@', regularSeason['trail_third_eff'], regularSeason['trail_opp_third_eff'])
features['Home_Third_Def'] = np.where(
    regularSeason['at'] == '@', regularSeason['trail_opp_third_def'], regularSeason['trail_third_def'])
features['Away_Third_Def'] = np.where(
    regularSeason['at'] == '@', regularSeason['trail_third_def'], regularSeason['trail_opp_third_def'])
features['Home_Fourth_Eff'] = np.where(
    regularSeason['at'] == '@', regularSeason['trail_opp_fourth_eff'], regularSeason['trail_fourth_eff'])
features['Away_Fourth_Eff'] = np.where(
    regularSeason['at'] == '@', regularSeason['trail_fourth_eff'], regularSeason['trail_opp_fourth_eff'])
features['Home_Fourth_Def'] = np.where(
    regularSeason['at'] == '@', regularSeason['trail_opp_fourth_def'], regularSeason['trail_fourth_def'])
features['Away_Fourth_Def'] = np.where(
    regularSeason['at'] == '@', regularSeason['trail_fourth_def'], regularSeason['trail_opp_fourth_def'])

features.to_csv('./cleaned.csv')
