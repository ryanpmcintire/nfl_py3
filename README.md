# nfl_py3
My first end-to-end machine learning project: Predicting nfl games with data scraped from pro-football-reference.com

To use this: 
1) Run build_dataset.py with start_year = 2009
2) Run nfl_cleane.R script to format data correctly
3) nfl_SVM.R will predict the outright winner of a game but, without the spread ~64% accurate
4) nfl_MLP.R will predict the outright winner of a game against the Vegas spread ~55% accurate

The below graph is the accuracy of the MLP on the test set

![alt text](https://github.com/ryanpmcintire/nfl_py3/blob/master/MLP_Accuracy.png)

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/2.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/2.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/2.0/">Creative Commons Attribution-NonCommercial-ShareAlike 2.0 Generic License</a>.
