import adaTree
import sys
from datetime import datetime

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Invalid number of parameters.')
        exit()

    week = sys.argv[1]
    year = datetime.now().year
    if not week:
        print('Week required. Please provide week number')
        exit()
    
    try:
        year = int(sys.argv[2])
    except:
        year = datetime.now().year
        print(f'Defaulting to {year}')
        

    
    df = adaTree.predict(week, year)
    df = df[['Away_Team','Home_Team','Home_Vegas_Spread','Predicted Spread','pick']]
    print(df)