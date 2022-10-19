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
from machines.learner_methods import print_eval, get_splits, read_data, plot, read_path, x_cols
from cleaner import clean


# toggles
showRegularSeasonDf = True


def showIf(data):
    if showRegularSeasonDf:
        dtale.show(data, subprocess=False)


def run_grid_search(year, week):
    X_train, X_test, y_train, y_test = get_splits(year, week)

    gridSearchResultPath = Path('gridSearchResults/adaTree.csv')
    gridSearchResultPath.parent.mkdir(parents=True, exist_ok=True)

    parameters = {
        'base_estimator__criterion': ['squared_error'],
        'base_estimator__max_depth': [i for i in range(6, 15, 1)],
        # 'base_estimator__min_samples_split': [2, 10, 20],
        # 'base_estimator__min_samples_leaf': [i for i in range(1, 10, 1)],
        'base_estimator__max_features': [6, 8, 10, 12, 15, 18, 21, 25],
        'n_estimators': [i for i in range(500, 3000, 500)],
        'learning_rate': [0.0001, 0.0005, 0.001],
        'loss': ['square'],
        'random_state': [88],
    }

    ada = AdaBoostRegressor(base_estimator=DecisionTreeRegressor())

    clf = GridSearchCV(
        ada, parameters, verbose=3, scoring=['neg_root_mean_squared_error', 'r2', 'neg_mean_absolute_error', 'neg_median_absolute_error'], refit='r2', n_jobs=16
    )
    estimator = clf.fit(X_train, y_train)
    print(clf.cv_results_.keys())
    resultDf = pd.concat(
        [
            pd.DataFrame(clf.cv_results_['params']),
            pd.DataFrame(clf.cv_results_['mean_test_neg_root_mean_squared_error']),
            pd.DataFrame(clf.cv_results_['std_test_neg_root_mean_squared_error']),
            pd.DataFrame(clf.cv_results_['rank_test_neg_root_mean_squared_error']),
            pd.DataFrame(clf.cv_results_['mean_test_r2']),
            pd.DataFrame(clf.cv_results_['std_test_r2']),
            pd.DataFrame(clf.cv_results_['rank_test_r2']),
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
    X_train, X_test, y_train, y_test = get_splits(year, week)
    data = read_data(read_path)

    predictionResultPath = Path(f'predictions/adaTree_week_{week}.csv')
    predictionResultPath.parent.mkdir(parents=True, exist_ok=True)

    regr = make_pipeline(
        AdaBoostRegressor(
            DecisionTreeRegressor(max_depth=4, max_features=12),
            n_estimators=500,
            learning_rate=0.001,
            loss='square',
            random_state=88,
        )
    )
    # These params outperform slightly on validation but seem to do slightly worse on test
    # regr = make_pipeline(AdaBoostRegressor(DecisionTreeRegressor(
    #     max_depth=13, max_features='sqrt', min_samples_leaf=5, min_samples_split=2), n_estimators=1750, learning_rate=0.0001, loss='linear', random_state=88))

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

def ada_test(year=2022, week=6):
    # clean(year, week)
    X_train, X_test, y_train, y_test = get_splits(year, week)


    regr = make_pipeline(
        AdaBoostRegressor(
            DecisionTreeRegressor(max_depth=6, max_features=21, criterion="friedman_mse"),
            n_estimators=500, 
            learning_rate=0.0005,
            loss='square',
            random_state=88,
        )
    )
    estimator = regr.fit(X_train, y_train)
    y_pred = estimator.predict(X_test)

    print_eval(X_test, y_pred, y_test)

"""
If running as main from modules folder:
some_python_environment -m machines.adaTree
"""
if __name__ == '__main__':
    ada_test()
    # run_grid_search(2022, 6)
