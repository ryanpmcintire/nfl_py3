from machines import adaTree
from cleaner import clean
from new_week import new_week
from datetime import datetime
import argparse

if __name__ == '__main__':
    parsed = argparse.ArgumentParser()
    parsed.add_argument('-y', '--year', type=int, default=datetime.now().year)
    parsed.add_argument('-w', '--week', type=int, default=1)
    args = parsed.parse_args()
    week, year = args.week, args.year
    df = new_week(year, week)
    clean(year, week)
    df = adaTree.predict(week, year)
    df = df[['Away_Team', 'Home_Team', 'Home_Vegas_Spread', 'Predicted Spread', 'pick']]
    print(df)
