# This script uses Multi-Layer Perceptron (MLP) to predict winner of an NFL game against the spread.
# MLP is a type of supervised feed-forward neural network that uses back-propagation for
# the training process. https://wikipedia.org/wiki/Multilayer_perceptron
#
# In the nfl_SVM script, it can be seen that the outright winner of NFL games can be
# predicted with some accuracy, using only statistics from previous games, and no
# information about rosters, player injuries etc.
#
# Now we will see if we can predict which team wins against the Vegas spread.
#
# It is documented that to achieve profitability on sports betting, a gambler must achieve
# a win rate in excess of 52.4%.
# When compared against Vegas bookmakers, the MLP created here consistently achieves accuracy
# greater than 55%, and in some seasons this number was as high as 60%.

# Libs
require(purrr)
require(tidyr)
require(dplyr)
# Caret for pre-processing stuff
require(caret)
# Mxnet is a decent package for creating MLPs
# However, the results of the network I've configured below will not always
# be the same, even when seeds are set. This is because I have used Xavier initializer.
# If you attempt to recreate this project, you should expect to see some small
# variations from my results here.
require(mxnet)

# Global flags to control model training and new week
# Flag to rerun model or not.
trainFlag <- FALSE

# Import features from cleane.R script
source("nfl_cleane.R")
source("nflFunctions.R")

# List of columns we want to keep
keep_cols <- c("Year", "Week", "Home_Team", "Home_Actual_Spread", "Home_Vegas_Spread",
               "Trail_Home_Score", "Trail_Away_Score", "Home_Allowed",
               "Away_Allowed", "Home_Pass_Eff", "Home_TO", "Away_TO",
               "Home_FTO", "Away_FTO", "Away_Pass_Eff", "Home_Pass_Def",
               "Away_Pass_Def", "Home_Rush_Eff","Away_Rush_Eff", "Home_Rush_Def",
               "Away_Rush_Def", "Home_Pen_Yds", "Away_Pen_Yds", "Home_Pen_Yds_Agg",
               "Away_Pen_Yds_Agg", "Home_Third_Eff", "Away_Third_Eff", "Home_Third_Def",
               "Away_Third_Def", "Home_Fourth_Eff", "Away_Fourth_Eff",
               "Home_Fourth_Def","Away_Fourth_Def")
# List of columns that track spreads, game identifying info
game_cols <- c("Year", "Week","Home_Team", "Home_Actual_Spread", "Home_Vegas_Spread")

# Split the features into two sets
# mlpFeatures will be fed into the training
# We'll store the spread_cols for later
# By doing this separation here, we won't have to worry about rows being misaligned
# down the road when/if we try to rejoin things
mlpFeatures <- as.data.frame(features)
mlpFeatures <- mlpFeatures[!duplicated(mlpFeatures), ]
# spreads <- as.data.frame(mlpFeatures[, spread_cols])
mlpFeatures <- mlpFeatures[, keep_cols]

# Create label column that tracks Vegas' performance against the actual final result.
# let V be vegas spread and A be actual:
# if |V| < |A| then Vegas correctly picked the favorite but underestimated the win margin
# if |V| = |A| then Vegas exactly predicted the outcome, the game is a push
# else, Vegas was incorrect and the underdog covered the spread.
# We can divide this into a binary classification as follows:
# Let F be the favorite wins (covers the spread) or the game is a push
# Let U be the underdog covers the spread or wins outright
# It is convenient to group the favorite winning with pushes as this gives us
# approximately equal number of each class
mlpFeatures$Pick <- as.factor(with(mlpFeatures,
                         ifelse(sign(Home_Vegas_Spread) == sign(Home_Actual_Spread) & abs(Home_Vegas_Spread) < abs(Home_Actual_Spread), "F",
                         ifelse(sign(Home_Vegas_Spread) == sign(Home_Actual_Spread) & abs(Home_Vegas_Spread) > abs(Home_Actual_Spread), "U",
                         ifelse(sign(Home_Vegas_Spread) < sign(Home_Actual_Spread), "U",
                         ifelse(sign(Home_Vegas_Spread) > sign(Home_Actual_Spread), "U", "F"))))))
# Take a quick look at the pick distribution to see if it passes sanity test:
# Should be close to 50/50 split between two classes.
summary(mlpFeatures$Pick)

# Split the data into a training and test set, remove year and week cols as we won't need them after this
# Depending on whether doing classification or regression, need to use different label columns
regLabel <- c(1:4)
classLabel <- c(1:3, 34)

train <- as.data.frame(mlpFeatures) %>%
  filter(Year > 2009 & Year < 2018)
train <- train[,-regLabel]

test <- as.data.frame(mlpFeatures) %>%
  filter(Year == 2018)
test <- test[,-regLabel]

# Set seed - (my results won't be perfectly reproducible because Xavier init)
mx.set.seed(42)

# Normalization is absolutely necessary
preProcValues <- preProcess(train, method = c("center", "scale"))
trainTransformed <- predict(preProcValues, train)
testTransformed <- predict(preProcValues, test)

# Convert the data into a matrix
trainMat <- data.matrix(trainTransformed)
testMat <- data.matrix(testTransformed)

# Set number of epochs
# With the parameters listed here, you should see that overfitting happens quickly
# The best performing epoch is usually within the first 15
# Some notes about the network config:
# Adam optimizier is key to getting this MLP to train quickly (less than 1 minute on my CPU)
# Only one hidden layer of 9 nodes is necessary. Two layers can also work but I found
# that the results were more variable between runs.
# Xavier init ultimately means the network can train on fewer epochs
epochs = 30
if(trainFlag == TRUE) {
nflMlp <- mx.mlp(trainMat[, -30], trainMat[, 30], hidden_node = c(9), out_node = 3,  initializer=mx.init.Xavier(rnd_type = "gaussian"),
                     num.round = epochs, array.batch.size = 8, learning.rate = 0.0005, eval.metric=mx.metric.accuracy,
                     activation = "relu", array.layout = "rowmajor", epoch.end.callback = mx.callback.save.checkpoint("checkpoint"),
                     optimizer = "adam", out_activation = "softmax") }
# dataset/label column/# of epochs
MLPEval(testMat, 30, epochs)
# 55%+ accuracy is easily achievable here.
# Also keep in mind that we've grouped the push result with the favorite result.
# This should mean our accuracy is slightly off. I think this is not in our favor
# and that the accuracy is underestimated by a very small margin.

# Save the best checkpoint
chckPt = 10
# mx.model.save(mx.model.load("checkpoint", j), "nfl_Class_Model_V2", chckPt)
# Load in a saved model
nfl_Class_Model_V2 <- mx.model.load("nfl_Class_Model_V2", chckPt)

# Combine everything into one data frame so we can view the games, spreads and the
# networks predictions for upcoming games side-by-side.
pred <- predict(nfl_Class_Model_V2, testMat[, -30], array.layout = "rowmajor")
pred.odds <- as.data.frame(t(pred))
finalResults <- as.data.frame(mlpFeatures[, game_cols]) %>%
  filter(Year == 2018)
finalResults$Pick <- ifelse(max.col(pred.odds) == 2, "Favorite/Push", "Underdog")
finalResults$FavOdds <- pred.odds[,2]
finalResults$DogOdds <- pred.odds[,3]