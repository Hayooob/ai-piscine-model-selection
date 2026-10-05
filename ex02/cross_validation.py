"""Exercise 2: cross validate a pipeline with cross_validate.

A single train/test split gives one score, which may be lucky or unlucky.
The cross validation trains the model on 10 different splits: the mean says
how good the model is, the standard deviation says how stable it is.
"""
# imports
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# data
housing = fetch_california_housing()
X, y = housing['data'], housing['target']
# split data train test
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.1,
                                                    shuffle=True,
                                                    random_state=43)
# pipeline
pipeline = [('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('lr', LinearRegression())]
pipe = Pipeline(pipeline)

# the cross validation runs on the train set only, the test set is untouched
cv_results = cross_validate(pipe, X_train, y_train, cv=10,
                            scoring="r2", return_train_score=True)
scores = cv_results["test_score"]

if __name__ == "__main__":
    print("Scores on validation sets:\n", scores, end="\n\n")
    print("Mean of scores on validation sets:\n", scores.mean(), end="\n\n")
    print("Standard deviation of scores on validation sets:\n", scores.std(),
          end="\n\n")
    print("The scores barely move from one fold to the other (std of 0.02): "
          "the model is stable, a first sign that it is not overfitted. An "
          "average r2 of 60% is a start, to be improved.")
