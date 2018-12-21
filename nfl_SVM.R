# This script uses support vector machines to predict winners of nfl games straight up. Spread is ignored

# Libs
require(purrr)
require(tidyr)
require(dplyr)
# I like this SVM package a lot. It takes minimal effort to get SVM's up and running
require(e1071)
# Caret for pre-processing stuff
require(caret)

# We'll take the SVMFeatures from our nfl_cleane.R and operate on them a bit more so they work nicely with SVM
source("nfl_cleane.R")
SVMFeatures <- features

# Game identifying columns
game_cols <- c("Home_Win", "Home_Team", "Home_Score", "Away_Score", "Year", "Week")
ratios <- as.data.frame(SVMFeatures[,game_cols])

# Convert game stats into ratios with home in numerator, away in denominator. I think this should offer
# small performance improvement versus keeping home & away cols separate but haven't definitively confirmed.
# These ratio columns represent the performance of the home & away teams compared to each other. I think
# this simplifies the number crunching for the SVM but, again I have not confirmed. One thing to keep in mind
# is that by organizing the data this way, we lose the ability to feature select on each home & away column.
# It could be the case that certain features are relevant to the away team and not the home team
# and vice versa. I don't care enough to confirm this here because this is just a proof of concept that
# we can in fact predict games with some level of accuracy. We can feature select in more advanced models
# (that incorporate spreads) later on if need be.
ratios$Points <- SVMFeatures$Trail_Home_Score / SVMFeatures$Trail_Away_Score
ratios$Allowed <- SVMFeatures$Home_Allowed / SVMFeatures$Away_Allowed
ratios$TO <- SVMFeatures$Home_TO / SVMFeatures$Away_TO
ratios$FTO <- SVMFeatures$Home_FTO / SVMFeatures$Away_FTO
ratios$Pass_Eff <- SVMFeatures$Home_Pass_Eff / SVMFeatures$Away_Pass_Eff
ratios$Pass_Def <- SVMFeatures$Home_Pass_Def / SVMFeatures$Away_Pass_Def
ratios$Rush_Eff <- SVMFeatures$Home_Rush_Eff / SVMFeatures$Away_Rush_Eff
ratios$Rush_Def <- SVMFeatures$Home_Rush_Def / SVMFeatures$Away_Rush_Def
ratios$PenYds <- SVMFeatures$Home_Pen_Yds / SVMFeatures$Away_Pen_Yds
ratios$PenYdsAgg <- SVMFeatures$Home_Pen_Yds_Agg / SVMFeatures$Away_Pen_Yds_Agg
ratios$ThirdEff <- SVMFeatures$Home_Third_Eff / SVMFeatures$Away_Third_Eff
ratios$ThirdDef <- SVMFeatures$Home_Third_Def / SVMFeatures$Away_Third_Def
ratios$FourthEff <- SVMFeatures$Home_Fourth_Eff / SVMFeatures$Away_Fourth_Eff
ratios$FourthDef <- SVMFeatures$Home_Fourth_Def / SVMFeatures$Away_Fourth_Def

# Replacing infs with 1. Not ideal, need to look into lapalce transform
ratios$FourthEff[is.infinite(ratios$FourthEff)] <- 1
ratios$FourthDef[is.infinite(ratios$FourthDef)] <- 1

# We get duplicates of every game entry because two teams, so remove one of them. Doesn't matter which one.
ratios <- ratios[!duplicated(ratios[, -(1:8)]), ]
# Remove home score and away score
ratios <- ratios[, -c(3:4)]
# Set seed for reproducibility
set.seed(42)

# Create training set with years before 2018. We'll use 2018 as test set
train <- ratios %>%
  filter(Year < 2018)
test <- ratios %>%
  filter(Year == 2018)

# Do some basic pre-processing. I tried lots of combinations of other pre-process
# techniques but didn't see any difference in performance
# Important to use the same preProcValues for all data sets
preProcValues <- preProcess(ratios, method = c("center", "scale"))
trainTransformed <- predict(preProcValues, train)
testTransformed <- predict(preProcValues, test)

# Save 10% of the training data to use for validation set
# If seed wasn't set this will be different each time
# Home Wins happens more frequently than losses
# This means we should organize our partitions so that Home_Win outcome is represented
# equally percentage-wise. Otherwise we open up possibility of the model over-favoring
# one outcome more or less than it should
train_ind <- createDataPartition(trainTransformed$Home_Win, p = 0.9, list = FALSE)
trainSample <- trainTransformed[train_ind, ]
validSample <- trainTransformed[-train_ind, ]

# Use a grid search to find best parameters, kernal etc.
# This can take some time if grid increments are small
# Start with linear kernel because it is least time consuming to tune
# Home team wins just under 57% of the time from 2009-2017 seasons. This is a good benchmark
# for the SVM to beat
tuned <- tune.svm(Home_Win~., data = trainSample, gamma = 2^seq(-15, -6, 1),  kernel = "linear")
summary(tuned)
# ~64.7% accuracy with this. Beating our benchmark but nothing to get too excited about
svmLinear <- svm(Home_Win~., data = trainSample, cost = 0.1, kernel = "linear")
# Using the svm, predict on the validation set
pred <- predict(svmLinear, validSample)
# Calculate number of games predicted correctly
record <- pred == validSample$Home_Win
sum(record == TRUE) / NROW(record)

# Functionalize the above performance check for convenience
# Takes in the model and dataset and returns the accuracy of the model
svmPerformance <- function(model, data) {
  pred <- predict(model, data)
  record <- pred == data$Home_Win
  sum(record == TRUE) / NROW(record)
}

# According to this paper: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.141.880&rep=rep1&type=pdf
# a radial kernal should be at least as good as linear. So if we want to squeeze some more performance
# out of our SVM, this seems like the natural next step
# I don't want to waste too much time tuning paramters. It can be helpful to visualize the error curves to figure out
# what range of parameters the error is minimized. This gives us a good idea for the bounds of the grid search
# Grid search with both hyper-params
tuned <- tune.svm(Home_Win~., data = trainSample, cost = 2^seq(-3, 1, 1), gamma = 2^seq(-6, 0, 1), kernel = "radial")
plot(tuned)
# Based on above grid search, we haven't narrowed down the cost much but we know gamma < 0.4 looks good
# You can continue to repeat the search with more granular increments on a more narrow grid and this should save
# time versus just brute forcing every possible combination

# With these params we can match the ~64.7% accuracy of the linear kernal but not able to do any better.
# It could still be possible to out-perform linear kernal with radial but don't want to spend too much
# time on the svm's
svmRadial <- svm(Home_Win~., data = trainSample, gamma = 0.03, cost = 0.5, kernel = "radial")
svmPerformance(svmRadial, validSample)

# Sigmoid kernel is also worth a shot
tuned <- tune.svm(Home_Win~., data = trainSample, cost = 2^seq(-3, 3, 1), gamma = 2^seq(-6, 0, 1), kernel = "sigmoid")
plot(tuned)
# From above plot, keeping gamma < 0.4 and cost < 8 appear to be nice boundaries

# With these params, Sigmoid kernel gives us a slight boost to accuracy on the validation set with ~65.7%
svmSigmoid <- svm(Home_Win~., data = trainSample, cost = 6.5, gamma = .01, kernel = "sigmoid")
svmPerformance(svmSigmoid, validSample)

# Finally, we'll compare the performance of these three SVM's against the test set
svmPerformance(svmLinear, testTransformed)
svmPerformance(svmRadial, testTransformed)
svmPerformance(svmSigmoid, testTransformed)
# Linear kernel performs best on test set with ~64.2%, then radial with ~63.1%, last is sigmoid with ~60.2%
# The linear kernel is the easiest to tune and performs the best in general
