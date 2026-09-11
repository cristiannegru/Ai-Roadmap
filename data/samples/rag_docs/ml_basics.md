# ML Basics

Supervised learning maps inputs to labels. Regression predicts continuous
house prices with linear models and mean squared error. Classification
predicts survived or churned with logistic regression and cross entropy.

Overfitting means memorizing training noise: training accuracy climbs while
validation accuracy falls. Prevent it with cross validation splits,
regularization penalties like ridge and lasso, and early stopping.

Feature engineering matters most: impute missing ages with the median,
one hot encode embarked ports, and standardize fares before fitting.
Evaluate classifiers with precision recall f1 and confusion matrices.
