import numpy as np
import sklearn
import xgboost as xgb

import dtale
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import AdaBoostRegressor

from sklearn.tree import DecisionTreeRegressor
from pathlib import Path
from learner_methods import print_eval, get_splits, plot


def grid_search():
    X_train, X_test, y_train, y_test = get_splits(2021, 5)
    xgb_model = xgb.XGBRegressor()
    parameters = {
        'max_depth': range(3, 15, 1),
        'n_estimators': range(100, 3000, 100),
        'learning_rate': [0.1, 0.01, 0.05, 0.001],
        'subsample': np.arange(0.1, 1, 0.1),
        'colsample_bytree': [0.3, 0.5, 0.7, 0.9, 1],
        'tree_method': ['exact', 'approx', 'hist'],
        'grow_policy': ['depthwise', 'lossguide'],
        'objective': ['reg:squarederror', 'reg:pseudohubererror']
    }

def xgb_test():
    # n_estimators=120000, gamma=0.001, objective='reg:squarederror', subsample=0.8, learning_rate=0.00002, tree_method="approx", max_depth=2, booster='gbtree', eval_metric='rmse', random_state=88
    xgb_model = xgb.XGBRegressor(n_estimators=140000, gamma=0.001, objective='reg:squarederror', subsample=0.85, learning_rate=0.000015, tree_method="approx", max_depth=3, booster='gbtree', eval_metric='rmse', random_state=88)
    X_train, X_test, y_train, y_test = get_splits(2021, 5)
    estimator = xgb_model.fit(X_train, y_train)
    y_pred = estimator.predict(X_test)
    print_eval(X_test, y_pred, y_test)
    plot(X_test, y_pred, y_test)

if __name__ == '__main__':
  xgb_test()