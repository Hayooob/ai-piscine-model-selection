"""Exercise 3: a grid search validated by a real cross validation.

Yesterday the grid search was scored on a single split. Here `cv=5` scores
every combination on 5 validation sets, so the model chosen is the one that
is good on average, not the one that was lucky once.
"""
# imports
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split

# data
housing = fetch_california_housing()
X, y = housing['data'], housing['target']
# split data train test
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.1,
                                                    shuffle=True,
                                                    random_state=43)

parameters = {
    'n_estimators': [10, 50, 75],
    'max_depth': [4, 7, 10],
}

if __name__ == "__main__":
    # 1. the grid search is scored with the MSE. Scikit-learn always
    # maximizes the score, so it uses the opposite of the MSE
    rf = RandomForestRegressor(random_state=43)
    gridsearch = GridSearchCV(rf,
                              parameters,
                              cv=5,
                              n_jobs=-1,
                              scoring='neg_mean_squared_error')
    gridsearch.fit(X_train, y_train)

    # 2. the best model and what it scored on the validation sets
    print("best params: ", gridsearch.best_params_)
    print("best score (neg MSE):", gridsearch.best_score_)
    print(f"the MSE is ~{-gridsearch.best_score_:.2f}", end="\n\n")

    print("cv_results_:")
    results = pd.DataFrame(gridsearch.cv_results_)
    print(results[["params", "mean_test_score", "std_test_score",
                   "rank_test_score"]].to_string(), end="\n\n")

    # 3. the score of the best model on the test set, kept aside until now
    test_score = gridsearch.score(X_test, y_test)
    print("score on the test set (neg MSE):", test_score)
    print(f"the MSE on the test set is ~{-test_score:.2f}", end="\n\n")
    print("The score on the test set is close to the score on the validation "
          "sets: the model is not overfitted.")
