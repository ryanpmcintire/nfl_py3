from tabnanny import verbose
import numpy as np
import sklearn
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeRegressor
from pathlib import Path
from machines.learner_methods import print_eval, get_splits, read_data, plot, read_path, x_cols
import pandas as pd
from sklearn.metrics import mean_squared_error, explained_variance_score, r2_score, mean_absolute_error, median_absolute_error, mean_absolute_percentage_error
from cleaner import clean



def xgb_learner(year=2022, week=6):
    # n_estimators=120000, gamma=0.001, objective='reg:squarederror', subsample=0.8, learning_rate=0.00002, tree_method="approx", max_depth=2, booster='gbtree', eval_metric='rmse', random_state=88
    # n_estimators=1500, objective='reg:squarederror', subsample=0.65, learning_rate=0.0015, tree_method="approx", max_depth=3, booster='gbtree', eval_metric='rmse', random_state=88
    clean(year, week)
    xgb_model = xgb.XGBRegressor(n_estimators=25000, objective='reg:squarederror', subsample=0.6, learning_rate=0.00017, tree_method="approx", max_depth=1, booster='gbtree', random_state=88)
    X_train, X_test, y_train, y_test = get_splits(year, week, test_size=.5, stratify='Home_Fav')

    estimator = xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
    y_pred = estimator.predict(X_test)
    print_eval(X_test, y_pred, y_test)
    plot(X_test, y_pred, y_test)

def predict(year, week):
    xgb_model = xgb.XGBRegressor(n_estimators=25000, objective='reg:squarederror', subsample=0.6, learning_rate=0.00017, tree_method="approx", max_depth=1, booster='gbtree', eval_metric='rmse', random_state=88)
    X_train, X_test, y_train, y_test = get_splits(year, week, test_size=.5, stratify='Home_Fav')
    data = read_data(read_path)

    predictionResultPath = Path(f'predictions/xgboost_week_{week}.csv')
    predictionResultPath.parent.mkdir(parents=True, exist_ok=True)

    estimator = xgb_model.fit(X_train, y_train)
    y_pred = estimator.predict(X_test)
    print_eval(X_test, y_pred, y_test)
    
    train = data[data['year'] < year]
    X_train = train[x_cols]
    y_train = train['Home_Actual_Spread']

    test = data[data['year'] > year - 1][data['week'] == week]
    X_test = test[x_cols]

    estimator.fit(X_train, y_train)
    y_test_pred = pd.DataFrame(estimator.predict(X_test), columns=['Predicted Spread'])

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
    predict(2022, 7)
    # xgb_learner()