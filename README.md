# nfl_py3
My first end-to-end machine learning project: Predicting nfl games with data scraped from pro-football-reference.com

To use this: 
1) Run build_dataset.py with start_year = 2009
2) Update path in nfl_cleane.r to reference the resulting .csv from step 1
3) To predict upcoming week, run up to and including line 77 in nfl_cleane.r - then create some dummy entries for upcoming week's game in the "nfl_newWeek.csv" with spreads from preferred source
4) Run the remaining lines of nfl_cleane.r
To-do: finish readme
