from sklearn.metrics import mean_squared_error, explained_variance_score, r2_score, mean_absolute_error, median_absolute_error, mean_absolute_percentage_error
from tabulate import tabulate
import pandas as pd
from pandas import DataFrame as df
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

read_path = './cleaned_with_rank.csv'

x_cols = [
    'week',
    'Home_Fav',
    'Home_Vegas_Spread',
    'Trail_Home_Score',
    'Trail_Away_Score',
    'Home_Allowed',
    'Away_Allowed',
    'Home_TO',
    'Away_TO',
    'Home_FTO',
    'Away_FTO',
    'Home_Pass_Eff',
    'Away_Pass_Eff',
    'Home_Pass_Def',
    'Away_Pass_Def',
    'Home_Rush_Eff',
    'Away_Rush_Eff',
    'Home_Rush_Def',
    'Away_Rush_Def',
    'Home_Pen_Yds',
    'Away_Pen_Yds',
    'Home_Pen_Yds_Agg',
    'Away_Pen_Yds_Agg',
    'Home_Third_Eff',
    'Away_Third_Eff',
    'Home_Third_Def',
    'Away_Third_Def',
    'Home_Fourth_Eff',
    'Away_Fourth_Eff',
    'Home_Fourth_Def',
    'Away_Fourth_Def',
    'wonLast',
    'lostLast',
    'lostLastAsFav',
    'wonLastAsDog',
    'Home_Off_Rank',
    'Home_Def_Rank',
    'Away_Off_Rank',
    'Away_Def_Rank'
]

x_cols_xgb = [
    'week',
    'Home_Fav',
    'Home_Vegas_Spread',
    'Home_Rush_Eff',
    'Away_Rush_Eff',
    'Home_Rush_Def',
    'Away_Rush_Def',
    'wonLast',
    'lostLast',
    'Home_Off_Rank',
    'Home_Def_Rank',
    'Away_Off_Rank',
    'Away_Def_Rank'
]

def print_eval(X_test, y_pred, y_test):
    metrics = {
        mean_squared_error: '<',
        # explained_variance_score: '>',
        r2_score: '>',
        mean_absolute_error: '<',
        median_absolute_error: '<',
        # mean_absolute_percentage_error: '<'
    }
    table = {}
    table[''] = ['Us: ', 'Vegas: ', 'Diff', 'Better']
    for metric, better in metrics.items():
        us = metric(y_test, y_pred)
        vegas = metric(y_test, X_test["Home_Vegas_Spread"])
        diff = us - vegas
        is_better = 'Yes' if eval(f'{diff}{better}0') else 'No'
        table[f'{metric.__name__} ({better} is better)'] = [us, vegas, diff, is_better]
    print(tabulate(table, headers='keys'))

def read_data(_read_path=read_path):
    data: df = pd.read_csv(_read_path)
    data = data.sort_values(['year', 'week'])
    # first 10 games for each team blank (10 x 32) but divide by 2 because we only record 1 instance of each matchup
    data = data.iloc[160:]
    return data

def get_splits(year, week=None, test_size = 0.2, stratify='week', _read_path=read_path):
    data = read_data(_read_path)

    train = data[data['year'] < year]
    if week:
        additional = data[(data['year'] == year) & (data['week'] < week)]
        train = pd.concat([train, additional])
        

    X = train[x_cols]
    y = train['Home_Actual_Spread']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=X[stratify], random_state=88
    )
    return X_train, X_test, y_train, y_test

def plot(X_test, y_pred, y_test):
    vegas = X_test['Home_Vegas_Spread'] - y_test
    vegas.sort_values(inplace=True)
    us = y_pred - y_test
    us.sort_values(inplace=True)
    x = np.arange(len(vegas))
    plt.scatter(x, vegas, label='vegas', s=0.5)
    plt.scatter(x, us, label='us', s=0.5)
    plt.legend()
    plt.savefig('error_compare.png')