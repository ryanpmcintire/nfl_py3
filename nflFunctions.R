# Useful functions used in the other scripts

# After preProcessing, data is in a format we cannot interpret. So we need to create
# a function to reverse the transformation so we can interpret model predictions after
# the model has been trained.
# This function takes in the preProcess values and the datasets and reverses
# the transformations.
revPredict <- function(preproc, data, digits=0) {
  data %>%
    select(one_of(preproc$mean %>% names)) %>%
    map2_df(preproc$std, ., function(sig, dat) dat * sig) %>%
    map2_df(preproc$mean, ., function(mu, dat) dat + mu)
}

# This function evaluates performance of stored checkpoints against a test set
# (Assumes that a model is still in memory)
# Give it the dataset and tell it which column is the label
# Also include number of epochs
MLPEval <- function(data, labelCol, epochs) {
  accuracyVector <- vector()
  for (i in 1:epochs) {
    # Load specific epoch
    checkpointModel <- mx.model.load("checkpoint", i)
    # Do prediction and and shrink label
    pred <- predict(checkpointModel, data[, -labelCol], array.layout = "rowmajor")
    pred.label <- max.col(t(pred)) -1
    # Wrap this in a try-catch because of quirk in mxnet that sets some early predictions = 0
    tryCatch({
      cfm <- confusionMatrix(as.factor(pred.label), as.factor(data[,labelCol]))},
      error = function(e){message(e)})
    # Update accuracy vector with each epoch's accuracy so we can visualize later on
    accuracyVector[i] <- ifelse(cfm$overall[1]>0, cfm$overall[1], 0)
    # Track best Accuracy
    print(paste("Epoch = ", i, "| Accuracy =", accuracyVector[i], "| Best =", max(accuracyVector)))
  }
    # Spit out a nice plot
  plot(accuracyVector,
       type = "o",
       main = "Test accuracy across epochs",
       xlab = "Epoch",
       ylab = "Accuracy",
       sub = "50% marker added for reference")
  abline(h = .50, col = "red")
}