import csv
import pandas as pd
from pandas import DataFrame as df, read_csv
import sys
import numpy as np
import re

game_span = 10
stopifnot2009 = 2009
currentSeasonYear = 2019
path = '/nfl_master_2009-2018.csv'
currentSeason = '/nfl_master_2019-2019.csv'
newWeekPath = 'C:/Users/Ryan/Downloads/nfl_newWeek16.csv'

rawSet: df = pd.read_csv(path)
newSet: df = pd.read_csv(currentSeason)

# newSet['week'] should be treated as a string

rawSet.append(newSet)

rawSet.rename({'boxscoreUri': 'time'}, axis=1, inplace=True)

playoffs = ['Wild Card', 'Division', 'Conf. Champ.', 'SuperBowl']
regularSeason: df = rawSet
regularSeason.drop(columns=playoffs)

if (len(regularSeason.columns) != 17):
    print('invalid number of weeks')
    sys.exit()

opponent_names = regularSeason['team'].unique()
opponent_names.append(pd.Series(['sdg', 'ram']), ignore_index=True)

verbose_names = regularSeason['verbose_name'].unique()
verbose_names.append(pd.Series(['Los Angeles Chargers', 'Los Angeles Rams']), ignore_index=True)
map = {'Los Angeles Chargers': 'sdg', 'Los Angeles Rams': 'ram'}
regularSeason['opponent'].map(lambda opp: map[opp] if opp in map else opp)

if(len(regularSeason['opponent'].unique()) != 32):
    print('invalid number of teams')
    sys.exit()

currentYear = regularSeason[regularSeason['year'] == currentSeasonYear]
currentYear.to_csv('nfl_newWeek.csv', index=False)

try:
    newWeek: df = pd.read_csv(newWeekPath)
    regularSeason = regularSeason[regularSeason['year'] < currentSeasonYear].append(newWeek)
except:
    print('new week file not found!')
#regularSeason.loc[regularSeason['at'] == '@', 'Home_Team'] = regularSeason['opponent']
#regularSeason.loc[regularSeason['at'] == '@', 'Away_Team'] = regularSeason['team']
regularSeason['Home_Team'] = np.where(regularSeason['at'] == '@', regularSeason['opponent'], regularSeason['team'])
regularSeason['Away_Team'] = np.where(regularSeason['at'] == '@', regularSeason['team'], regularSeason['opponent'])

regularSeason['Home_Score'] = np.where(regularSeason['at'] == '@', regularSeason['opp_score'], regularSeason['team_score'])
regularSeason['Away_Score'] = np.where(regularSeason['at'] == '@', regularSeason['team_score'], regularSeason['opp_score'])

pattern = '^(San |New |St. |Green |Tampa |Kansas |Los )*'
replace = ''
regularSeason['Vegas_Line_Close'].replace(to_replace=pattern, value=replace, inplace=True, regex=True)

regularSeason[['Fav_City', 'Fav_Team', 'Fav_Spread']] = regularSeason['Vegas_Line_Close'].str.split(pat=' ', expand=True)
#regularSeason['Fav_Spread'] = regularSeason['Vegas_Line_Close'].str.split(pat=' ', expand=True)[2]

if(regularSeason.loc[1, 'year'] == stopifnot2009):
    sys.exit()

longNames = regularSeason['Fav_Team'].unique()
shortNames = ['crd', 'jax', 'sea', 'nyg', 'chi', 'oti', 'min', 'atl',
             'sdg', 'nor', 'kan', 'sfo', 'ram', 'den', 'car', 'dal',
             'was', 'pit', 'rav', 'phi', 'cin', 'nwe', 'gnb', 'nyj',
             'det', 'tam', 'htx', '000', 'mia', 'buf', 'rai', 'clt', 'cle']
shortNameLookup = df({'shortName': shortNames, 'longName': longNames})
regularSeason['favorite'] = regularSeason['Fav_Team'].transform(lambda name: shortNameLookup[shortNameLookup['longName'] == name]['shortName'])

regularSeason['HomeFav'] = (regularSeason['favorite'] == '000').map({True: 0, False: -1}) 
regularSeason.loc[regularSeason['favorite'] == regularSeason['Home_Team'], 'HomeFav'] = 1
regularSeason['underdog'] = (regularSeason['favorite'] == regularSeason['team']).map({True: 0, False: 1})

regularSeason.fillna(0)

regularSeason['Home_Vegas_Spread'] = regularSeason[regularSeason['HomeFav'] == 1]['Fav_Spread']
regularSeason['Home_Vegas_Spread'] = np.where(regularSeason['HomeFav'] == 0, 0, regularSeason['Fav_Spread'].transform(lambda fav: fav * -1))
regularSeason['Home_Actual_Spread'] = -1 * (regularSeason['Home_Score'] - regularSeason['Away_Score'])

away_pass_stats = regularSeason[['aCmp', 'aAtt', 'aYd', 'aTD', 'aINT']]
home_pass_stats = regularSeason[['hCmp', 'hAtt', 'hYd', 'hTD', 'hINT']]

away_rush_stats = regularSeason[['aRush', 'aRYds', 'aRTDs']]
home_rush_stats = regularSeason[['hRush', 'hRYds', 'hRTDs']]
away_pen_yds = regularSeason[['aPen', 'aPenYds']]
home_pen_yds = regularSeason[['hPen', 'hPenYds']]
away_third_downs = regularSeason[['aThrdConv', 'aThrd']]
home_third_downs = regularSeason[['hThrdConv', 'hThrd']]
away_fourth_downs = regularSeason[['aFrthConv', 'aFrth']]
home_fourth_downs = regularSeason[['hFrthConv', 'hFrth']]

@staticmethod
def assignStats(regularSeason: df, column: str):
    offensive = column
    defensive = f'd{column}'
    homestat = f'h{column}'
    awaystat = f'a{column}'
    regularSeason[offensive] = np.where(regularSeason['at'] == '@', regularSeason[awaystat], regularSeason[homestat])
    regularSeason[defensive] = np.where(regularSeason['at'] == '@', regularSeason[homestat], regularSeason[awaystat])

passing = ['Cmp', 'Att', 'Yd', 'TD', 'INT']
rushing = ['Rush', 'RYds', 'RTDs']
downs = ['ThrdConv', 'Thrds', 'FrthConv', 'Frths']

for column in passing:
    assignStats(regularSeason, column)
regularSeason['team_pass_eff'] = regularSeason['Yd'] / regularSeason['Att']
regularSeason['team_pass_def'] = regularSeason['dYd'] / regularSeason['dAtt']

for column in rushing:
    assignStats(regularSeason, column)
regularSeason['team_rush_eff'] = regularSeason['RYds'] / regularSeason['Rush']
regularSeason['team_rush_def'] = regularSeason['dRYds'] / regularSeason['dRush']

for column in downs:
    assignStats(regularSeason, column)
regularSeason['third_eff'] = regularSeason['ThrdConv'] / regularSeason['Thrds']
regularSeason['third_def'] = regularSeason['dThrdConv'] / regularSeason['dThrds']
regularSeason['fourth_eff'] = regularSeason['FrthConv'] / regularSeason['Frths']
regularSeason['fourth_def'] = regularSeason['dFrthConv'] / regularSeason['dFrths']

regularSeason['fourth_eff'] = regularSeason['fourth_eff'].fillna(0.0001)
regularSeason['fourth_def'] = regularSeason['fourth_def'].fillna(0.0001)

regularSeason['Pen'] = np.where(regularSeason['at'] == '@', regularSeason['aPen'], regularSeason['hPen'])
regularSeason['PenAgg'] = np.where(regularSeason['at'] == '@', regularSeason['hPen'], regularSeason['aPen'])
regularSeason['PenYds'] = np.where(regularSeason['at'] == '@', regularSeason['aPenYds'], regularSeason['hPenYds'])
regularSeason['PenYdsAgg'] = np.where(regularSeason['at'] == '@', regularSeason['hPenYds'], regularSeason['aPenYds'])

regularSeason['opponent_col'] = list(range(regularSeason['boxscoreUri'].count()))


grp = regularSeason.groupby(['team'])
## dont know if this works
regularSeason['lostLastAsFav'] = np.where(grp['favorite'].shift(1) == grp['team'].shift(1) & grp['result'] == 'L', 1, 0)
regularSeason['wonLastAsDog'] = np.where(grp['underdog'].shift(1) == 1 & grp['result'] == 'W', 1, 0)
##

@staticmethod
def rolling_averages(dataframe, column, key, game_span):
    dataframe[column] = dataframe.groupby(['team'])[key].rolling(game_span).mean().reset_index(0,drop=True)
    
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

@staticmethod
def get_opp_trail(dataframe, column, key):
    dataframe[column] = dataframe.loc[dataframe['opponent_col']][key]

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

features = regularSeason[['year', 'week', 'Home_Team', 'Away_Team']]
features['Home_Win'] = np.where(regularSeason['result'] == 'L' & regularSeason['at'] == '', 'L',
    np.where(regularSeason['result'] == 'W' & regularSeason['at'] == '', "W",
        np.where(regularSeason['result'] == 'L' & regularSeason['at'] == '@', 'W', 'L')
    )
)
features['Home_Fav'] = regularSeason['HomeFav']
features['Home_Vegas_Spread'] = regularSeason['Home_Vegas_Spread']
features['Home_Actual_Spread'] = regularSeason['Home_Actual_Spread']
features['Home_Score'] = regularSeason['Home_Score']
features['Away_Score'] = regularSeason['Away_Score']
features['Trail_Home_Score'] = np.where(regularSeason['at'] == '@', regularSeason['trail_opp_score'], regularSeason['trail_score'])
features['Trail_Away_Score'] = np.where(regularSeason['at'] == '@', regularSeason['trail_score'], regularSeason['trail_opp_score'])
features['Home_Allowed'] = np.where(regularSeason['at'] == '@', regularSeason['trail_opp_allow'], regularSeason['trail_allow'])
features['Away_Allowed'] = np.where(regularSeason['at'] == '@', regularSeason['trail_allow'], regularSeason['trail_opp_allow'])
features['Home_TO'] = np.where(regularSeason['at'] == '@', regularSeason['trail_opp_to'], regularSeason['trail_to'])
features['Away_TO'] = np.where(regularSeason['at'] == '@', regularSeason['trail_to'], regularSeason['trail_opp_to'])
features['Home_FTO'] = np.where(regularSeason['at'] == '@', regularSeason['trail_opp_fto'], regularSeason['trail_fto'])
features['Away_FTO'] = np.where(regularSeason['at'] == '@', regularSeason['trail_fto'], regularSeason['trail_opp_fto'])
features['Home_Pass_Eff'] = np.where(regularSeason['at'] == '@', regularSeason['trail_opp_pass_eff'], regularSeason['trail_pass_eff'])
features['Away_Pass_Eff'] = np.where(regularSeason['at'] == '@', regularSeason['trail_pass_eff'], regularSeason['trail_opp_pass_eff'])
features['Home_Pass_Def'] = np.where(regularSeason['at'] == '@', regularSeason['trail_opp_pass_def'], regularSeason['trail_pass_def'])
features['Away_Pass_Def'] = np.where(regularSeason['at'] == '@', regularSeason['trail_pass_def'], regularSeason['trail_opp_pass_def'])
features['Home_Rush_Eff'] = np.where(regularSeason['at'] == '@', regularSeason['trail_opp_rush_eff'], regularSeason['trail_rush_eff'])
features['Away_Rush_Eff'] = np.where(regularSeason['at'] == '@', regularSeason['trail_rush_eff'], regularSeason['trail_opp_rush_eff'])
features['Home_Rush_Def'] = np.where(regularSeason['at'] == '@', regularSeason['trail_opp_rush_def'], regularSeason['trail_rush_def'])
features['Away_Rush_Def'] = np.where(regularSeason['at'] == '@', regularSeason['trail_rush_def'], regularSeason['trail_opp_rush_def'])
features['Home_Pen_Yds'] = np.where(regularSeason['at'] == '@', regularSeason['trail_opp_penYds'], regularSeason['trail_penYds'])
features['Away_Pen_Yds'] = np.where(regularSeason['at'] == '@', regularSeason['trail_penYds'], regularSeason['trail_opp_penYds'])
features['Home_Pen_Yds_Agg'] = np.where(regularSeason['at'] == '@', regularSeason['trail_opp_penYdsAgg'], regularSeason['trail_penYdsAgg'])
features['Away_Pen_Yds_Agg'] = np.where(regularSeason['at'] == '@', regularSeason['trail_penYdsAgg'], regularSeason['trail_opp_penYdsAgg'])
features['Home_Third_Eff'] = np.where(regularSeason['at'] == '@', regularSeason['trail_opp_third_eff'], regularSeason['trail_third_eff'])
features['Away_Third_Eff'] = np.where(regularSeason['at'] == '@', regularSeason['trail_third_eff'], regularSeason['trail_opp_third_eff'])
features['Home_Third_Def'] = np.where(regularSeason['at'] == '@', regularSeason['trail_opp_third_def'], regularSeason['trail_third_def'])
features['Away_Third_Def'] = np.where(regularSeason['at'] == '@', regularSeason['trail_third_def'], regularSeason['trail_opp_third_def'])
features['Home_Fourth_Eff'] = np.where(regularSeason['at'] == '@', regularSeason['trail_opp_fourth_eff'], regularSeason['trail_fourth_eff'])
features['Away_Fourth_Eff'] = np.where(regularSeason['at'] == '@', regularSeason['trail_fourth_eff'], regularSeason['trail_opp_fourth_eff'])
features['Home_Fourth_Def'] = np.where(regularSeason['at'] == '@', regularSeason['trail_opp_fourth_def'], regularSeason['trail_fourth_def'])
features['Away_Fourth_Def'] = np.where(regularSeason['at'] == '@', regularSeason['trail_fourth_def'], regularSeason['trail_opp_fourth_def'])

