"""Exercise 1: split the data in k folds with KFold.

Each fold is used once as the validation set while the model learns on the
others. This is what cross_validate and GridSearchCV do internally.
"""
import numpy as np
from sklearn.model_selection import KFold

X = np.array(np.arange(1, 21).reshape(10, -1))
y = np.array(np.arange(1, 11))

kf = KFold(n_splits=5)

if __name__ == "__main__":
    for fold, (train_index, test_index) in enumerate(kf.split(X), start=1):
        print("Fold: ", fold)
        print("TRAIN:", train_index, "TEST:", test_index, end="\n\n")
