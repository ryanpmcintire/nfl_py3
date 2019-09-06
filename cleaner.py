import csv;
import pandas as pd;
from pandas import DataFrame as df, read_csv;
import sys;
import numpy as np;
import re;

game_span = 10;
stopifnot2009 = 2009;
currentSeasonYear = 2018;
path = 'c:/workspace/nfl_py3/nfl_master_2009-2017.csv';
currentSeason = 'c:/workspace/nfl_py3/nfl_master_2018-2018.csv';
newWeekPath = 'C:/Users/Ryan/Downloads/nfl_newWeek16.csv'

rawSet: df = pd.read_csv(path);
newSet: df = pd.read_csv(currentSeason);

# newSet['week'] should be treated as a string

rawSet.append(newSet);

rawSet.rename({'boxscoreUri': 'time'}, axis=1, inplace=True);

playoffs = ['Wild Card', 'Division', 'Conf. Champ.', 'SuperBowl'];
regularSeason: df = rawSet;
regularSeason.drop(columns=playoffs);

if (len(regularSeason.columns) != 17):
    print('invalid number of weeks')
    sys.exit();

opponent_names = regularSeason['team'].unique();
opponent_names.append(pd.Series(['sdg', 'ram']), ignore_index=True);

verbose_names = regularSeason['verbose_name'].unique();
verbose_name.append(pd.Series(['Los Angeles Chargers', 'Los Angeles Rams']), ignore_index=True);
map = {'Los Angeles Chargers': 'sdg', 'Los Angeles Rams': 'ram'};
regularSeason[opponent].map(lambda opp: map[opp] if opp in map else opp)

if(len(regularSeason[opponent].unique()) != 32):
    print('invalid number of teams');
    sys.exit();

currentYear = regularSeason[regularSeason['year'] == currentSeasonYear]
currentYear.to_csv('nfl_newWeek.csv', index=False);

newWeek: pf = pd.read_csv(newWeekPath)
regularSeason = regularSeason[regularSeason['year'] < currentSeasonYear].append(newWeek)

#regularSeason.loc[regularSeason['at'] == '@', 'Home_Team'] = regularSeason['opponent'];
#regularSeason.loc[regularSeason['at'] == '@', 'Away_Team'] = regularSeason['team'];
regularSeason['Home_Team'] = np.where(regularSeason['at'] == '@', regularSeason['opponent'], regularSeason['team']);
regularSeason['Away_Team'] = np.where(regularSeason['at'] == '@', regularSeason['team'], regularSeason['opponent']);

regularSeason['Home_Score'] = np.where(regularSeason['at'] == '@', regularSeason['opp_score'], regularSeason['team_score']);
regularSeason['Away_Score'] = np.where(regularSeason['at'] == '@', regularSeason['team_score'], regularSeason['opp_score']);

pattern = '^(San |New |St. |Green |Tampa |Kansas |Los )*';
replace = '';
regularSeason['Vegas_Line_Close'].replace(to_replace=pattern, value=replace, inplace=True, regex=True);

regularSeason[['Fav_City', 'Fav_Team', 'Fav_Spread']] = regularSeason['Vegas_Line_Close'].str.split(pat=' ', expand=True);
#regularSeason['Fav_Spread'] = regularSeason['Vegas_Line_Close'].str.split(pat=' ', expand=True)[2];

if(regularSeason.loc[1, 'year'] == stopifnot2009):
    sys.exit();

longNames = regularSeason['Fav_Team'].unique();
shortNames = ['crd', 'jax', 'sea', 'nyg', 'chi', 'oti', 'min', 'atl',
             'sdg', 'nor', 'kan', 'sfo', 'ram', 'den', 'car', 'dal',
             'was', 'pit', 'rav', 'phi', 'cin', 'nwe', 'gnb', 'nyj',
             'det', 'tam', 'htx', '000', 'mia', 'buf', 'rai', 'clt', 'cle'];
shortNameLookup = df({'shortName': shortNames, 'longName': longNames});
regularSeason['favorite'] = regularSeason['Fav_Team'].transform(lambda name: shortNameLookup[shortNameLookup['longName'] == name]['shortName']);

regularSeason['HomeFav'] = (regularSeason['favorite'] == '000').map({True: 0, False: -1}); 
regularSeason.loc[regularSeason['favorite'] == regularSeason['Home_Team'], 'HomeFav'] = 1;
regularSeason['underdog'] = (regularSeason['favorite'] == regularSeason['team']).map({True: 0, False: 1});

regularSeason.fillna(0);

regularSeason['Home_Vegas_Spread'] = regularSeason[regularSeason['HomeFav'] == 1]['Fav_Spread'];
regularSeason['Home_Vegas_Spread'] = np.where(regularSeason['HomeFav'] == 0, 0, regularSeason['Fav_Spread'].transform(lambda fav: fav * -1));
regularSeason['Home_Actual_Spread'] = -1 * (regularSeason['Home_Score'] - regularSeason['Away_Score']);

