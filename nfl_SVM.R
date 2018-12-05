# This script uses support vector machines to predict winners of nfl games straight up. Spread is ignored
# The goal here is to get some experience with SVMs but also prove that games can be predicted with
# some accuracy better than coin flip.

# Libs
require(purrr)
require(tidyr)
require(dplyr)
# Well known SVM package. It takes minimal effort to get SVM's up and running with this
require(e1071)
# Caret for pre-processing stuff. This plays nice with lots of other packages
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

# Create training set with years before 2018. Reserve 2018 for test set
train <- ratios %>%
  filter(Year < 2018)
test <- ratios %>%
  filter(Year == 2018 & Week < 13)

# Do some basic pre-processing. I tried lots of combinations of other pre-process
# techniques but didn't see much difference in performance
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
tuned <- tune.svm(Home_Win~., data = trainSample, cost = 2^seq(-15, 5, 1),  kernel = "linear")
summary(tuned)

# ~64.7% accuracy with this. Beating our benchmark but nothing to get too excited about
svmLinear <- svm(Home_Win~., data = trainSample, cost = 0.1, kernel = "linear")
pred <- predict(svmLinear, validSample)
# Calculate number of games predicted correctly
record <- pred == validSample$Home_Win
sum(record == TRUE) / NROW(record)

# According to this paper: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.141.880&rep=rep1&type=pdf
# a radial kernal should be at least as good as linear. So if we want to squeeze some more performance
# out of our SVM, this seems like the natural next step. It takes quite a bit more effort to tune parameters for
# radial kernal, though.