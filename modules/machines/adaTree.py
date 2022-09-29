import pandas as pd
import dtale
from pandas import DataFrame as df
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from pathlib import Path
import numpy as np

# toggles
showRegularSeasonDf = True
# week = 2
# year = 2022
readPath = './cleaned.csv'

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
]


def showIf(data):
    if showRegularSeasonDf:
        dtale.show(data, subprocess=False)


def read_data(readPath):
    data: df = pd.read_csv(readPath)
    data = data.sort_values(['year', 'week'])
    # first 10 games for each team blank (10 x 32) but divide by 2 because we only record 1 instance of each matchup
    data = data.iloc[160:]
    return data


def train_machine(year):
    data = read_data(readPath)

    train = data[data['year'] < year]
    X = train[x_cols]
    y = train['Home_Actual_Spread']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=88
    )
    return X_train, X_test, y_train, y_test


def run_grid_search(year):
    X_train, X_test, y_train, y_test = train_machine(year)

    gridSearchResultPath = Path('gridSearchResults/adaTree.csv')
    gridSearchResultPath.parent.mkdir(parents=True, exist_ok=True)

    parameters = {
        'base_estimator__criterion': ['squared_error'],
        'base_estimator__max_depth': [i for i in range(11, 17, 1)],
        'base_estimator__min_samples_split': [2, 10, 20],
        'base_estimator__min_samples_leaf': [i for i in range(1, 10, 1)],
        'base_estimator__max_features': ['sqrt'],
        'n_estimators': [i for i in range(1600, 2100, 25)],
        'learning_rate': [0.00005, 0.0001, 0.0002],
        'loss': ['linear'],  # , 'square', 'exponential'],
        'random_state': [88],
    }

    ada = AdaBoostRegressor(base_estimator=DecisionTreeRegressor())

    clf = GridSearchCV(
        ada, parameters, verbose=3, scoring='neg_root_mean_squared_error', n_jobs=16
    )
    estimator = clf.fit(X_train, y_train)
    resultDf = pd.concat(
        [
            pd.DataFrame(clf.cv_results_['params']),
            pd.DataFrame(clf.cv_results_['mean_test_score']),
        ],
        axis=1,
    )
    resultDf.to_csv(gridSearchResultPath)
    print('best estimator: ', clf.best_estimator_)
    print('best params: ', clf.best_params_)
    y_pred = estimator.predict(X_test)
    print("MSE: ", mean_squared_error(y_test, y_pred))


# After doing grid search, put best parameters here


def predict(week, year):
    X_train, X_test, y_train, y_test = train_machine(year)
    data = read_data(readPath)

    predictionResultPath = Path(f'predictions/adaTree_week_{week}.csv')
    predictionResultPath.parent.mkdir(parents=True, exist_ok=True)

    regr = make_pipeline(
        AdaBoostRegressor(
            DecisionTreeRegressor(max_depth=13, max_features='sqrt', min_samples_leaf=5, min_samples_split=2),
            n_estimators=1750,
            learning_rate=0.0001,
            loss='linear',
            random_state=88,
        )
    )
    # These params outperform slightly on validation but seem to do slightly worse on test
    # regr = make_pipeline(AdaBoostRegressor(DecisionTreeRegressor(
    #     max_depth=13, max_features='sqrt', min_samples_leaf=5, min_samples_split=2), n_estimators=1750, learning_rate=0.0001, loss='linear', random_state=88))

    # measure of Vegas' accuracy <- this is benchmark to beat
    vegas_accuracy = mean_squared_error(X_train['Home_Vegas_Spread'], y_train)
    print("Vegas MSE: ", vegas_accuracy)

    regr.fit(X_train, y_train)
    y_val_pred = regr.predict(X_test)
    our_accuracy = mean_squared_error(y_test, y_val_pred)
    print("Validation MSE: ", our_accuracy)
    print(f"Better than Vegas == {vegas_accuracy > our_accuracy}")

    train = data[data['year'] < year]
    X_train = train[x_cols]
    y_train = train['Home_Actual_Spread']

    test = data[data['year'] > year - 1][data['week'] == week]
    X_test = test[x_cols]

    regr.fit(X_train, y_train)
    y_test_pred = pd.DataFrame(regr.predict(X_test), columns=['Predicted Spread'])

    predictions = (
        test[['Home_Team', 'Away_Team', 'Home_Vegas_Spread']]
        .reset_index(drop=True)
        .join(y_test_pred['Predicted Spread'].reset_index(drop=True))
    )
    predictions['pick'] = np.where(
        predictions['Predicted Spread'] <= predictions['Home_Vegas_Spread'],
        predictions['Home_Team'],
        predictions['Away_Team'],
    )
    predictions.to_csv(predictionResultPath)
    return predictions


if __name__ == '__main__':
    run_grid_search(2021)
