from matplotlib.pyplot import axis
import sklearn
import pandas as pd
import dtale
from pandas import DataFrame as df
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from pathlib import Path
import numpy as np

# toggles
showRegularSeasonDf = True
runGridSearch = False
week = 1

readPath = './cleaned.csv'

data: df = pd.read_csv(readPath,)
data = data.sort_values(['year', 'week'])


def showIf():
    if showRegularSeasonDf:
        dtale.show(data, subprocess=False)


# first 10 games for each team blank (10 x 32) but divide by 2 because we only record 1 instance of each matchup
data = data.iloc[160:]

x_cols = ['Home_Fav', 'Home_Vegas_Spread', 'Trail_Home_Score', 'Trail_Away_Score', 'Home_Allowed',
          'Away_Allowed', 'Home_TO', 'Away_TO', 'Home_FTO', 'Away_FTO',
          'Home_Pass_Eff', 'Away_Pass_Eff', 'Home_Pass_Def', 'Away_Pass_Def',
          'Home_Rush_Eff', 'Away_Rush_Eff', 'Home_Rush_Def', 'Away_Rush_Def',
          'Home_Pen_Yds', 'Away_Pen_Yds', 'Home_Pen_Yds_Agg', 'Away_Pen_Yds_Agg',
          'Home_Third_Eff', 'Away_Third_Eff', 'Home_Third_Def', 'Away_Third_Def',
          'Home_Fourth_Eff', 'Away_Fourth_Eff', 'Home_Fourth_Def',
          'Away_Fourth_Def']

train = data[data['year'] < 2022]
X = train[x_cols]
y = train['Home_Actual_Spread']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=.2, random_state=88)


if runGridSearch:
    gridSearchResultPath = Path('gridSearchResults/adaTree.csv')
    gridSearchResultPath.parent.mkdir(parents=True, exist_ok=True)

    parameters = {
        'base_estimator__criterion': ['squared_error'],
        'base_estimator__max_depth': [15],
        # 'base_estimator__min_samples_split': [2, 10, 20],
        # 'base_estimator__min_samples_leaf': [1, 5, 10],
        'base_estimator__max_features': ['sqrt', 'log2'],
        'n_estimators': [i for i in range(1600, 2000, 25)],
        'learning_rate': [0.00001, 0.0001, 0.001],
        'loss': ['linear'],  # , 'square', 'exponential'],
        'random_state': [88]
    }

    ada = AdaBoostRegressor(base_estimator=DecisionTreeRegressor())

    clf = GridSearchCV(ada, parameters, verbose=3,
                       scoring='neg_root_mean_squared_error', n_jobs=12)
    estimator = clf.fit(X_train, y_train)
    resultDf = pd.concat([pd.DataFrame(clf.cv_results_['params']), pd.DataFrame(
        clf.cv_results_['mean_test_score'])], axis=1)
    resultDf.to_csv(gridSearchResultPath)
    print('best estimator: ', clf.best_estimator_)
    print('best params: ', clf.best_params_)
    y_pred = estimator.predict(X_test)
    print("MSE: ", mean_squared_error(y_test, y_pred))

# After doing grid search, put best parameters here
else:
    predictionResultPath = Path('predictions/adaTree_week_{}.csv'.format(week))
    predictionResultPath.parent.mkdir(parents=True, exist_ok=True)

    regr = make_pipeline(AdaBoostRegressor(DecisionTreeRegressor(
        max_depth=15, max_features='sqrt'), n_estimators=1800, learning_rate=0.0001, loss='linear', random_state=88))

    vegas_pred = X_train['Home_Vegas_Spread']
    # measure of Vegas' accuracy <- this is benchmark to beat
    vegas_accuracy = mean_squared_error(vegas_pred, y_train)
    print("Vegas MSE: ", vegas_accuracy)

    estimator = regr.fit(X_train, y_train)
    y_val_pred = regr.predict(X_test)
    our_accuracy = mean_squared_error(y_test, y_val_pred)
    print("Validation MSE: ", our_accuracy)
    print("Better than Vegas == {}".format(vegas_accuracy > our_accuracy))

    train = data[data['year'] < 2022]
    X_train = train[x_cols]
    y_train = train['Home_Actual_Spread']

    # ToDo add week check
    test = data[data['year'] > 2021]
    X_test = test[x_cols]

    estimator = regr.fit(X_train, y_train)
    y_test_pred = pd.DataFrame(regr.predict(
        X_test), columns=['Predicted Spread'])

    predictions = test[['Home_Team', 'Away_Team',
                        'Home_Vegas_Spread']].reset_index(drop=True).join(y_test_pred['Predicted Spread'].reset_index(drop=True))
    predictions['pick'] = np.where(predictions['Home_Vegas_Spread']
                                   > predictions['Predicted Spread'], 'home', 'away')
    predictions.to_csv(predictionResultPath)
