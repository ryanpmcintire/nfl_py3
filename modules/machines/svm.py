import sklearn
import pandas as pd
import dtale
from pandas import DataFrame as df
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

showRegularSeasonDf = True


def showIf():
    if showRegularSeasonDf:
        dtale.show(data, subprocess=False)


path = './cleaned.csv'

data: df = pd.read_csv(
    path,
)
data = data.sort_values(['year', 'week'])

# first 10 games for each team blank (10 x 32) but divide by 2 because we only record 1 instance of each matchup
data = data.iloc[160:]

x_cols = [
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
]
print(data.columns)

train = data[data['year'] < 2022]
X = train[x_cols]
y = train['Home_Actual_Spread']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=88
)

regr = make_pipeline(SVR(C=1.0, epsilon=0.2))
estimator = regr.fit(X_train, y_train)
y_pred = estimator.predict(X_test)
print("MSE: ", mean_squared_error(y_test, y_pred))
