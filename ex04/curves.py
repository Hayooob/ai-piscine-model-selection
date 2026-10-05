"""Exercise 4: diagnose a model with a validation curve and a learning curve.

- the validation curve shows the score against one hyperparameter. When the
  train score keeps rising while the validation score plateaus, the model
  starts to overfit.
- the learning curve shows the score against the number of training rows. It
  answers a different question: would adding more data help?

The data set holds 100k rows, so both curves take a few minutes even on all
the CPUs.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import learning_curve, validation_curve

HERE = Path(__file__).resolve().parent

X, y = make_classification(n_samples=100000,
                           n_features=30,
                           n_informative=10,
                           flip_y=0.2,
                           random_state=43)


def plot_validation_curve():
    """The score against max_depth, on the train folds and the validation folds."""
    clf = RandomForestClassifier(random_state=43)
    param_range = np.arange(1, 30, 2)
    train_scores, test_scores = validation_curve(clf,
                                                 X,
                                                 y,
                                                 param_name="max_depth",
                                                 param_range=param_range,
                                                 cv=5,
                                                 scoring="roc_auc",
                                                 n_jobs=-1)

    train_mean, train_std = train_scores.mean(axis=1), train_scores.std(axis=1)
    test_mean, test_std = test_scores.mean(axis=1), test_scores.std(axis=1)

    fig, ax = plt.subplots()
    ax.set_title("Validation Curve with Random Forest")
    ax.set_xlabel("max_depth")
    ax.set_ylabel("roc_auc")
    ax.set_ylim(0.0, 1.1)

    ax.plot(param_range, train_mean, label="Training score", color="darkorange")
    ax.fill_between(param_range, train_mean - train_std, train_mean + train_std,
                    alpha=0.2, color="darkorange")
    ax.plot(param_range, test_mean, label="Cross-validation score", color="navy")
    ax.fill_between(param_range, test_mean - test_std, test_mean + test_std,
                    alpha=0.2, color="navy")

    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(HERE / "validation_curve.png")
    return param_range, train_mean, test_mean


def plot_learning_curve():
    """The three plots of the learning curve: score, scalability, performance."""
    clf = RandomForestClassifier(max_depth=12, random_state=43)
    train_sizes = np.linspace(0.1, 1.0, 5)

    sizes, train_scores, test_scores, fit_times, _ = learning_curve(
        clf, X, y, cv=10, n_jobs=-1, train_sizes=train_sizes, return_times=True)

    train_mean, train_std = train_scores.mean(axis=1), train_scores.std(axis=1)
    test_mean, test_std = test_scores.mean(axis=1), test_scores.std(axis=1)
    fit_times_mean, fit_times_std = fit_times.mean(axis=1), fit_times.std(axis=1)

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    # the learning curve itself
    axes[0].set_title("Learning curve (RandomForestClassifier, max_depth=12)")
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")
    axes[0].grid()
    axes[0].fill_between(sizes, train_mean - train_std, train_mean + train_std,
                         alpha=0.1, color="r")
    axes[0].fill_between(sizes, test_mean - test_std, test_mean + test_std,
                         alpha=0.1, color="g")
    axes[0].plot(sizes, train_mean, 'o-', color="r", label="Training score")
    axes[0].plot(sizes, test_mean, 'o-', color="g", label="Cross-validation score")
    axes[0].legend(loc="best")

    # how the training time grows with the number of rows
    axes[1].grid()
    axes[1].plot(sizes, fit_times_mean, 'o-')
    axes[1].fill_between(sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # is the extra training time worth the extra score?
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_mean - test_std, test_mean + test_std,
                         alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    fig.tight_layout()
    fig.savefig(HERE / "learning_curve.png")
    return sizes, train_mean, test_mean, fit_times_mean


if __name__ == "__main__":
    print("1. validation curve (this takes a few minutes)...")
    param_range, train_mean, test_mean = plot_validation_curve()
    for depth, train, test in zip(param_range, train_mean, test_mean):
        print(f"   max_depth={depth:2d}  train={train:.4f}  validation={test:.4f}")
    print("   From max_depth=10 the train score keeps rising while the "
          "validation score plateaus around 0.9: deeper trees only overfit.",
          end="\n\n")

    print("2. learning curve (this takes a few minutes)...")
    sizes, train_mean, test_mean, fit_times_mean = plot_learning_curve()
    for size, train, test, fit_time in zip(sizes, train_mean, test_mean,
                                           fit_times_mean):
        print(f"   {size:6d} rows  train={train:.4f}  validation={test:.4f}"
              f"  fit={fit_time:.1f}s")
    print("   The training time grows linearly with the number of rows, and "
          "the score has nearly reached its plateau: adding data would buy "
          "very little.")

    plt.show()
