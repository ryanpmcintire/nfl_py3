# nfl_py3
My first end-to-end machine learning project: Predicting nfl games with data scraped from pro-football-reference.com

To use this: 
1) Run build_dataset.py with start_year = 2009
2) Run nfl_cleane.R script to format data correctly
3) nfl_SVM.R will predict the outright winner of a game but, without the spread ~64% accurate
4) nfl_MLP.R will predict the outright winner of a game against the Vegas spread ~55% accurate
Graph of test accuracy from nfl_MLP.R

