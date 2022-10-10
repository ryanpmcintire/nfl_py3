import pandas as pd
import dtale
from pandas import DataFrame as df
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from pathlib import Path
import numpy as np
from machines.learner_methods import print_eval, get_splits, read_data, plot

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


def run_grid_search(year, week):
    X_train, X_test, y_train, y_test = get_splits(year, week)

    gridSearchResultPath = Path('gridSearchResults/adaTree.csv')
    gridSearchResultPath.parent.mkdir(parents=True, exist_ok=True)

    parameters = {
        'base_estimator__criterion': ['squared_error', 'friedman_mse'],
        'base_estimator__max_depth': [i for i in range(3, 15, 1)],
        # 'base_estimator__min_samples_split': [2, 10, 20],
        # 'base_estimator__min_samples_leaf': [i for i in range(1, 10, 1)],
        'base_estimator__max_features': ['sqrt'],
        'n_estimators': [i for i in range(500, 3000, 500)],
        'learning_rate': [0.00005, 0.0001, 0.0002],
        'loss': ['square'],
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
    print_eval(X_test, y_pred, y_test)


# After doing grid search, put best parameters here


def predict(week, year):
    X_train, X_test, y_train, y_test = get_splits(year - 1, week)
    data = read_data(readPath)

    predictionResultPath = Path(f'predictions/adaTree_week_{week}.csv')
    predictionResultPath.parent.mkdir(parents=True, exist_ok=True)

    regr = make_pipeline(
        AdaBoostRegressor(
            DecisionTreeRegressor(max_depth=15, max_features='sqrt'),
            n_estimators=2025,
            learning_rate=0.0001,
            loss='square',
            random_state=88,
        )
    )
    # These params outperform slightly on validation but seem to do slightly worse on test
    # regr = make_pipeline(AdaBoostRegressor(DecisionTreeRegressor(
    #     max_depth=13, max_features='sqrt', min_samples_leaf=5, min_samples_split=2), n_estimators=1750, learning_rate=0.0001, loss='linear', random_state=88))

    # measure of Vegas' accuracy <- this is benchmark to beat

    estimator = regr.fit(X_train, y_train)
    y_pred = estimator.predict(X_test)
    print_eval(X_test, y_pred, y_test)

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

def ada_test():
    X_train, X_test, y_train, y_test = get_splits(2021, 5)
    regr = make_pipeline(
        AdaBoostRegressor(
            DecisionTreeRegressor(max_depth=13, max_features='sqrt', min_samples_leaf=5, min_samples_split=2),
            n_estimators=1750,
            learning_rate=0.0001,
            loss='square',
            random_state=88,
        )
    )
    estimator = regr.fit(X_train, y_train)
    y_pred = estimator.predict(X_test)
    print_eval(X_test, y_pred, y_test)

if __name__ == '__main__':
    # ada_test()
    run_grid_search(2021, 5)
