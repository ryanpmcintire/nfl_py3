from machines import adaTree, xgboost
from cleaner import clean
from new_week import new_week
from build_dataset import add_new_week
from datetime import datetime
from pathlib import Path
import argparse
import config
import pandas as pd
import numpy as np
from rank import Rank

if __name__ == '__main__':
    parsed = argparse.ArgumentParser()
    parsed.add_argument('-y', '--year', type=int, default=datetime.now().year)
    parsed.add_argument('-w', '--week', type=int, default=1)
    args = parsed.parse_args()
    week, year = args.week, args.year
    
    if week != 1:
        master: pd.DataFrame = pd.read_csv(config.MASTER_PATH)
        is_previous_week_empty = (master[(master['year'] == year) & (master['week'] == week-1)]).empty
        if is_previous_week_empty:
            add_new_week(week-1,2012,year)

    df = new_week(year, week)
    clean(year, week)
    Rank().hits_test() # Update cleaned.csv with ranks

    # Ada
    df: pd.DataFrame = adaTree.predict(year=year, week=week)
    adaPicks = df[['Away_Team', 'Home_Team', 'Home_Vegas_Spread', 'Predicted Spread', 'pick']].rename(columns={'Predicted Spread': 'Ada Spread', 'pick': 'Ada Pick'})

    # Xgb
    df: pd.DataFrame = xgboost.predict(year=year, week=week)
    xgboostPicks = df[['Away_Team', 'Home_Team', 'Home_Vegas_Spread', 'Predicted Spread', 'pick']].rename(columns={'Predicted Spread': 'XgBoost Spread', 'pick': 'XgBoost Pick'})
    
    # Merge picks
    df = adaPicks.merge(xgboostPicks, on=['Away_Team', 'Home_Team', 'Home_Vegas_Spread'])

    # Compare and average picks
    df['Same Pick?'] = np.where(df['Ada Pick'] == df['XgBoost Pick'], 'Yes', 'No')
    df['Average Predicted Spread'] = (df['Ada Spread'] + df['XgBoost Spread']) / 2
    df['Average Pick'] = np.where(df['Average Predicted Spread'] <= df['Home_Vegas_Spread'], df['Home_Team'], df['Away_Team'])

    # Rearrange columns for readability
    df = df[['Away_Team', 'Home_Team', 'Home_Vegas_Spread', 'Ada Spread', 'XgBoost Spread', 'Average Predicted Spread', 'Ada Pick', 'XgBoost Pick', 'Same Pick?', 'Average Pick']]

    predictionResultPath = Path(f'predictions/combined_picks_{week}.md')
    predictionResultPath.parent.mkdir(parents=True, exist_ok=True)
    df.to_markdown(predictionResultPath)

    print(df)
