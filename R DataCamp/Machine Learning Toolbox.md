In-sample RMSE for linear regression on diamonds
===

As you saw in the video, included in the course is the `diamonds`dataset, which is a classic dataset from the `ggplot2` package. The dataset contains physical attributes of diamonds as well as the price they sold for. One interesting modeling challenge is predicting diamond price based on their attributes using something like a linear regression.

Recall that to fit a linear regression, you use the `lm()` function in the following format:

```R
mod <- lm(y ~ x, my_data)
```

To make predictions using `mod` on the original data, you call the `predict()` function:

```R
pred <- predict(mod, my_data)
```

Code:

```R
# Fit lm model: model
model <- lm(price ~ .,diamonds)

# Predict on full data: p
p <- predict(model, diamonds)

# Compute errors: error
error <- p - diamonds$price

# Calculate RMSE
sqrt(mean(error^2))
[1] 1129.843
```

Randomly order the data frame
===

One way you can take a train/test split of a dataset is to order the dataset randomly, then divide it into the two sets. This ensures that the training set and test set are both random samples and that any biases in the ordering of the dataset (e.g. if it had originally been ordered by price or size) are not retained in the samples we take for training and testing your models. You can think of this like shuffling a brand new deck of playing cards before dealing hands.

First, you set a random seed so that your work is reproducible and you get the same random split each time you run your script:

```
set.seed(42)
```

Next, you use the `sample()` function to shuffle the row indices of the `diamonds` dataset. You can later use these these indices to reorder the dataset.

```
rows <- sample(nrow(diamonds))
```

Finally, you can use this random vector to reorder the diamonds dataset:

```
diamonds <- diamonds[rows, ]
```

Code:

```R
# Set seed
set.seed(42)

# Shuffle row indices: rows
rows <- sample(nrow(diamonds))

# Randomly order data
diamonds <- diamonds[rows,]
```

Try an 80/20 split
===

Now that your dataset is randomly ordered, you can split the first 80% of it into a training set, and the last 20% into a test set. You can do this by choosing a split point approximately 80% of the way through your data:

```
split <- round(nrow(mydata) * .80)
```

You can then use this point to break off the first 80% of the dataset as a training set:

```
mydata[1:split, ]
```

And then you can use that same point to determine the test set:

```
mydata[(split + 1):nrow(mydata), ]
```

Code: 

```R
# Determine row to split on: split
split <- round(nrow(diamonds) * .8)

# Create train
train <- diamonds[1:split,]

# Create test
test <- diamonds[(split+1):nrow(diamonds),]
```

Predict on test set
===

Now that you have a randomly split training set and test set, you can use the `lm()` function as you did in the first exercise to fit a model to your training set, rather than the entire dataset. Recall that you can use the formula interface to the linear regression function to fit a model with a specified target variable using all other variables in the dataset as predictors:

```
mod <- lm(y ~ ., training_data)
```

You can use the `predict()` function to make predictions from that model on new data. The new dataset must have all of the columns from the training data, but they can be in a different order with different values. Here, rather than re-predicting on the training set, you can predict on the test set, which you did not use for training the model. This will allow you to determine the out-of-sample error for the model in the next exercise:

```
p <- predict(model, new_data)
```

Code:

```R
# Fit lm model on train: model
model <- lm(price ~ ., train)

# Predict on test: p
p <- predict(model, test)
```

Calculate test set RMSE by hand
===

Now that you have predictions on the test set, you can use these predictions to calculate an error metric (in this case RMSE) on the test set and see how the model performs out-of-sample, rather than in-sample as you did in the first exercise. You first do this by calculating the errors between the predicted diamond prices and the actual diamond prices by subtracting the predictions from the actual values.

Once you have an error vector, calculating RMSE is as simple as squaring it, taking the mean, then taking the square root:

```R
sqrt(mean(error^2))
```

Code:

```R
# Compute errors: error
error <- test$price - p

# Calculate RMSE
sqrt(mean(error^2))
```

10-fold cross-validation
===

A better approach to validating models is to use multiple systematic test sets, rather than a single random train/test split. Fortunately, the `caret` package makes this very easy to do:

```R
model <- train(y ~ ., my_data)
```

`caret` supports many types of cross-validation, and you can specify which type of cross-validation and the number of cross-validation folds with the `trainControl()` function, which you pass to the `trControl`argument in `train()`:

```R
model <- train(
  y ~ ., my_data,
  method = "lm",
  trControl = trainControl(
    method = "cv", number = 10,
    verboseIter = TRUE
  )
)
```

It's important to note that you pass the method for modeling to the main `train()` function and the method for cross-validation to the `trainControl()` function.

```R
# Fit lm model using 10-fold CV: model
model <- train(
  price ~ ., diamonds,
  method = "lm",
  trControl = trainControl(
    method = "cv", number = 10,
    verboseIter = TRUE
  )
)

# Print model to console
print(model)

Linear Regression 

25000 samples
    9 predictor

No pre-processing
Resampling: Cross-Validated (10 fold) 
Summary of sample sizes: 22500, 22500, 22500, 22500, 22501, 22501, ... 
Resampling results:

  RMSE     Rsquared 
  1138.53  0.9195797

Tuning parameter 'intercept' was held constant at a value of TRUE
```

5-fold cross-validation
===

In this course, you will use a wide variety of datasets to explore the full flexibility of the `caret` package. Here, you will use the famous Boston housing dataset, where the goal is to predict median home values in various Boston suburbs.

```R
# Fit lm model using 5-fold CV: model
model <- train(
  medv ~ ., Boston,
  method = "lm",
  trControl = trainControl(
    method = "cv", number = 5,
    verboseIter = TRUE
  )
)

# Print model to console
print(model)

Linear Regression 

506 samples
 13 predictor

No pre-processing
Resampling: Cross-Validated (5 fold) 
Summary of sample sizes: 404, 405, 407, 405, 403 
Resampling results:

  RMSE      Rsquared 
  4.783064  0.7349697

Tuning parameter 'intercept' was held constant at a value of TRUE
```

You can do more than just one iteration of cross-validation. Repeated cross-validation gives you a better estimate of the test-set error. You can also repeat the entire cross-validation procedure. This takes longer, but gives you many more out-of-sample datasets to look at and much more precise assessments of how well the model performs.

```R
# Fit lm model using 5 x 5-fold CV: model
model <- train(
  medv ~ ., Boston,
  method = "lm",
  trControl = trainControl(
    method = "cv", number = 5,
    repeats = 5, verboseIter = TRUE
  )
)

# Print model to console
print(model)

Linear Regression 

506 samples
 13 predictor

No pre-processing
Resampling: Cross-Validated (5 fold) 
Summary of sample sizes: 406, 405, 404, 406, 403 
Resampling results:

  RMSE      Rsquared 
  4.916463  0.7220276

Tuning parameter 'intercept' was held constant at a value of TRUE
```

Making predictions on new data
===

Finally, the model you fit with the `train()` function has the exact same `predict()` interface as the linear regression models you fit earlier.

After fitting a model with `train()`, you can simply call `predict()`with new data, e.g:

```R
predict(my_model, new_data)
```

---

Fit a logistic regression model
===

Once you have your random training and test sets you can fit a logistic regression model to your training set using the `glm()` function. `glm()` is a more advanced version of `lm()` that allows for more varied types of regression models, aside from plain vanilla ordinary least squares regression.

Be sure to pass the argument `family = "binomial"` to `glm()` to specify that you want to do logistic (rather than linear) regression. For example:

```R
glm(Target ~ ., family = "binomial", dataset)
```

Don't worry about warnings like `glm.fit: algorithm did not converge` or `glm.fit: fitted probabilities numerically 0 or 1 occurred`. These are common on smaller datasets and usually don't cause any issues. They typically mean your dataset is *perfectly seperable*, which can cause problems for the math behind the model, but R's `glm()`function is almost always robust enough to handle this case with no problems.

Once you have a `glm()` model fit to your dataset, you can predict the outcome (e.g. rock or mine) on the `test` set using the `predict()`function with the argument `type = "response"`:

```R
predict(my_model, test, type = "response")
```

Calculate a confusion matrix
===

A confusion matrix is a very useful tool for calibrating the output of a model and examining all possible outcomes of your predictions (true positive, true negative, false positive, false negative).

Before you make your confusion matrix, you need to "cut" your predicted probabilities at a given threshold to turn probabilities into class predictions. You can do this easily with the `ifelse()` function, e.g.:

```R
class_prediction <-
  ifelse(probability_prediction > 0.50,
         "positive_class",
         "negative_class"
  )
```

You could make such a contingency table with the `table()` function in base R, but `confusionMatrix()` in `caret` yields a lot of useful ancillary statistics in addition to the base rates in the table. You can calculate the confusion matrix (and the associated statistics) using the predicted outcomes as well as the actual outcomes, e.g.:

```R
confusionMatrix(predicted, actual)
```

Code:

```R
# Calculate class probabilities: p_class
p_class <- ifelse(p > 0.5, "M", "R")

# Create confusion matrix
confusionMatrix(p_class, test$Class)

Confusion Matrix and Statistics

          Reference
Prediction  M  R
         M 40 17
         R  8 18
                                          
               Accuracy : 0.6988          
                 95% CI : (0.5882, 0.7947)
    No Information Rate : 0.5783          
    P-Value [Acc > NIR] : 0.01616         
                                          
                  Kappa : 0.3602          
 Mcnemar's Test P-Value : 0.10960         
True positive rate Sensitivity : 0.8333          
True negative rate Specificity : 0.5143          
         Pos Pred Value : 0.7018          
         Neg Pred Value : 0.6923          
             Prevalence : 0.5783          
         Detection Rate : 0.4819          
   Detection Prevalence : 0.6867          
      Balanced Accuracy : 0.6738          
                                          
       'Positive' Class : M
```

Try another threshold
===

In the previous exercises, you used a threshold of 0.50 to cut your predicted probabilities to make class predictions (rock vs mine). However, this classification threshold does not always align with the goals for a given modeling problem.

For example, pretend you want to identify the objects you are really certain are mines. In this case, you might want to use a probability threshold of 0.90 to get **fewer predicted mines, but with greater confidence in each prediction.**

In this exercise, you will simply look at the highly likely mines, which you can isolate using the `ifelse()` function in R:

```
pred <- ifelse(probability > threshold, "M", "R")
```

You can then call the `confusionMatrix()` function in the same way as in the previous exercise.

```
confusionMatrix(pred, actual)
```

Code:

```R
# Apply threshold of 0.9: p_class
p_class <- ifelse(p > .9, "M","R")

# Create confusion matrix
confusionMatrix(p_class, test$Class)

Confusion Matrix and Statistics

          Reference
Prediction  M  R
         M 40 15
         R  8 20
                                          
               Accuracy : 0.7229          
                 95% CI : (0.6138, 0.8155)
    No Information Rate : 0.5783          
    P-Value [Acc > NIR] : 0.004583        
                                          
                  Kappa : 0.416           
 Mcnemar's Test P-Value : 0.210903        
                                          
            Sensitivity : 0.8333          
            Specificity : 0.5714          
         Pos Pred Value : 0.7273          
         Neg Pred Value : 0.7143          
             Prevalence : 0.5783          
         Detection Rate : 0.4819          
   Detection Prevalence : 0.6627          
      Balanced Accuracy : 0.7024          
                                          
       'Positive' Class : M
```

From probabilites to confusion matrix
===

Conversely, say you want to be really certain that your model correctly identifies all the mines as mines. In this case, you might use a prediction threshold of 0.10, instead of 0.90.

You can construct the confusion matrix in the same way you did before, using your new predicted classes:

```R
pred <- ifelse(probability > threshold, "M", "R")
```

You can then call the `confusionMatrix()` function in the same way as in the previous exercise:

```R
confusionMatrix(pred, actual)
```

```R
# Apply threshold of 0.10: p_class
p_class <- ifelse(p > .1, "M", "R")

# Create confusion matrix
confusionMatrix(p_class, test$Class)
Confusion Matrix and Statistics

          Reference
Prediction  M  R
         M 40 18
         R  8 17
                                          
               Accuracy : 0.6867          
                 95% CI : (0.5756, 0.7841)
    No Information Rate : 0.5783          
    P-Value [Acc > NIR] : 0.02806         
                                          
                  Kappa : 0.3319          
 Mcnemar's Test P-Value : 0.07756         
                                          
            Sensitivity : 0.8333          
            Specificity : 0.4857          
         Pos Pred Value : 0.6897          
         Neg Pred Value : 0.6800          
             Prevalence : 0.5783          
         Detection Rate : 0.4819          
   Detection Prevalence : 0.6988          
      Balanced Accuracy : 0.6595          
                                          
       'Positive' Class : M
```

Plot an ROC curve
===

As you saw in the video, an ROC curve is a really useful shortcut for summarizing the performance of a classifier over all possible thresholds. This saves you a lot of tedious work computing class predictions for many different thresholds and examining the confusion matrix for each.

My favorite package for computing ROC curves is **caTools**, which contains a function called **colAUC()**. This function is very user-friendly and can actually calculate ROC curves for multiple predictors at once. In this case, you only need to calculate the ROC curve for one predictor, e.g.:

```R
colAUC(predicted_probabilities, actual, plotROC = TRUE)
```

The function will return a score called AUC (more on that later) and the `plotROC = TRUE` argument will return the plot of the ROC curve for visual inspection.

```R
# Predict on test: p
p <- predict(model, test, type = "response")

# Make ROC curve
colAUC(p, test$Class, plotROC = TRUE)
             [,1]
M vs. R 0.7452381
```

Customizing trainControl
===

As you saw in the video, area under the ROC curve is a very useful, single-number summary of a model's ability to discriminate the positive from the negative class (e.g. mines from rocks). An AUC of 0.5 is no better than random guessing, an AUC of 1.0 is a perfectly predictive model, and an AUC of 0.0 is perfectly anti-predictive (which rarely happens).

This is often a much more useful metric than simply ranking models by their accuracy at a set threshold, as different models might require different calibration steps (looking at a confusion matrix at each step) to find the optimal classification threshold for that model.

You can use the `trainControl()` function in `caret` to use AUC (instead of acccuracy), to tune the parameters of your models. The `twoClassSummary()` convenience function allows you to do this easily.

When using `twoClassSummary()`, be sure to always include the argument `classProbs = TRUE` or your model will throw an error! (You cannot calculate AUC with just class predictions. You need to have class probabilities as well.)

```R
# Create trainControl object: myControl
myControl <- trainControl(
  method = "cv",
  number = 10,
  summaryFunction = twoClassSummary,
  classProbs = TRUE, # IMPORTANT!
  verboseIter = TRUE
)
```

Using custom trainControl
===

Now that you have a custom `trainControl` object, it's easy to fit `caret` models that use AUC rather than accuracy to tune and evaluate the model. You can just pass your custom `trainControl` object to the `train()` function via the `trControl` argument, e.g.:

```R
train(<standard arguments here>, trControl = myControl)
```

This syntax gives you a convenient way to store a lot of custom modeling parameters and then use them across multiple different calls to `train()`.

```R
# Train glm with custom trainControl: model
model <- train(Class ~ ., Sonar, method = "glm", trControl = myControl)


# Print model to console
print(model)
Generalized Linear Model 

208 samples
 60 predictor
  2 classes: 'M', 'R' 

No pre-processing
Resampling: Cross-Validated (10 fold) 
Summary of sample sizes: 187, 187, 188, 187, 187, 187, ... 
Resampling results:

  ROC        Sens      Spec     
  0.7397222  0.755303  0.6711111
```

Fit a random forest
===

As you saw in the video, random forest models are much more flexible than linear models, and can model complicated nonlinear effects as well as automatically capture interactions between variables. They tend to give very good results on real world data, so let's try one out on the wine quality dataset, where the goal is to predict the human-evaluated quality of a batch of wine, given some of the machine-measured chemical and physical properties of that batch.

Fitting a random forest model is exactly the same as fitting a generalized linear regression model, as you did in the previous chapter. You simply change the `method` argument in the `train` function to be `"ranger"`. The `ranger` package is a rewrite of R's classic `randomForest` package and fits models much faster, but gives almost exactly the same results. We suggest that all beginners use the `ranger` package for random forest modeling.

```R
# Fit random forest: model
model <- train(
  quality ~ .,
  tuneLength = 1,
  data = wine, method = "ranger",
  trControl = trainControl(method = "cv", number = 5, verboseIter = TRUE)
)

# Print model to console
print(model)
Random Forest 

100 samples
 12 predictor

No pre-processing
Resampling: Cross-Validated (5 fold) 
Summary of sample sizes: 79, 80, 80, 81, 80 
Resampling results:

  RMSE       Rsquared 
  0.6361579  0.3734367

Tuning parameter 'mtry' was held constant at a value of 3
```

Try a longer tune length
===

Recall from the video that random forest models have a primary tuning parameter of `mtry`, which controls how many variables are exposed to the splitting search routine at each split. For example, suppose that a tree has a total of 10 splits and `mtry = 2`. This means that there are 10 samples of 2 predictors each time a split is evaluated.

Use a larger tuning grid this time, but stick to the defaults provided by the `train()` function. Try a `tuneLength` of 3, rather than 1, to explore some more potential models, and plot the resulting model using the `plot` function.

```R
# Fit random forest: model
model <- train(
  quality ~ .,
  tuneLength = 3,
  data = wine, method = "ranger",
  trControl = trainControl(method = "cv", number = 5, verboseIter = TRUE)
)

# Print model to console
print(model)

# Plot model
plot(model)
Random Forest 

100 samples
 12 predictor

No pre-processing
Resampling: Cross-Validated (5 fold) 
Summary of sample sizes: 79, 81, 81, 79, 80 
Resampling results across tuning parameters:

  mtry  RMSE       Rsquared 
   2    0.6851657  0.1793044
   7    0.6871065  0.1899214
  12    0.6835114  0.2114151

RMSE was used to select the optimal model using  the smallest value.
The final value used for the model was mtry = 12.
```

Fit a random forest with custom tuning
===

Now that you've explored the default tuning grids provided by the `train()` function, let's customize your models a bit more.

You can provide any number of values for `mtry`, from 2 up to the number of columns in the dataset. In practice, there are diminishing returns for much larger values of `mtry`, so you will use a custom tuning grid that explores 2 simple models (`mtry = 2` and `mtry = 3`) as well as one more complicated model (`mtry = 7`).

```R
# Fit random forest: model
model <- train(
  quality ~ .,
  tuneGrid = data.frame(mtry = c(2, 3, 7)),
  data = wine, method = "ranger",
  trControl = trainControl(method = "cv", number = 5, verboseIter = TRUE)
)

# Print model to console
print(model)

# Plot model
plot(model)

100 samples
 12 predictor

No pre-processing
Resampling: Cross-Validated (5 fold) 
Summary of sample sizes: 80, 80, 79, 80, 81 
Resampling results across tuning parameters:

  mtry  RMSE       Rsquared 
  2     0.6548705  0.3194206
  3     0.6548308  0.3012394
  7     0.6442832  0.3087214

RMSE was used to select the optimal model using  the smallest value.
The final value used for the model was mtry = 7.
```

