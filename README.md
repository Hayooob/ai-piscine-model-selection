# Model selection

Solutions for the Model selection day of the AI piscine: the k-fold, the
cross validation, a grid search validated on several folds, and the two
curves that tell whether a model underfits or overfits.

## Structure

```
.
├── ex00/
│   ├── requirements.txt
│   └── Notebook_ex00.ipynb
├── ex01/k_fold.py
├── ex02/cross_validation.py
├── ex03/gridsearch_cv.py
└── ex04/curves.py            -> validation_curve.png, learning_curve.png
```

## Virtual environment (ex00)

```bash
python -m venv env/ex00
source env/ex00/Scripts/activate      # Windows (Git Bash); Linux/macOS: source env/ex00/bin/activate
pip install -r ex00/requirements.txt
python --version
python -c "import jupyter, numpy, pandas, matplotlib, sklearn"
```

Tested with Python 3.13, scikit-learn 1.9.1, pandas 3.0.6, NumPy 2.5.3 and
matplotlib 3.11.2.

## Running an exercise

```bash
python ex03/gridsearch_cv.py
```

ex01 and ex02 are instant, ex03 takes around 20 seconds. ex04 works on
100000 rows and takes several minutes even on all the CPUs.

## Why cross validate

A single train/test split gives a single score. That score may be good by
luck, and a model that looks best on that one split is not necessarily the
best model. The cross validation cuts the train set in k folds, trains the
model k times and keeps every score: the **mean** says how good the model
is, the **standard deviation** says how stable it is. Only then does the
test set, untouched until the end, give the final score.

## Results

- **ex01** — the 5 folds match the expected output: each pair of rows is
  used once as the validation set.
- **ex02** — the 10 scores stay between 0.546 and 0.625, for a mean of
  `0.60201392526743` and a standard deviation of `0.0215`. The model behaves
  the same on every fold, so it is stable, which is a first sign that it is
  not overfitted.
- **ex03** — best parameters `{'max_depth': 10, 'n_estimators': 75}` with a
  MSE of **0.29** on the validation sets and **0.27** on the test set. The
  two being close means the selected model generalizes.
- **ex04** — see below.

## Reading the two curves of ex04

The **validation curve** plots the score against `max_depth`. The training
score keeps climbing towards 1 as the trees get deeper, because a deep tree
can memorize the training rows; the cross-validation score reaches a plateau
around 0.9 from `max_depth = 10`. Everything after that point is overfitting:
the model gets better at the data it has seen and no better at the rest.
This is why it is worth plotting for the hyperparameters that control the
overfitting.

The **learning curve** answers another question: would more data help? It
plots the score against the number of training rows, for a fixed
`max_depth=12`. The two other plots that come with it show the training time
against the number of rows (it grows linearly here) and the score against
the training time, which is what tells whether more data is worth the extra
training time.
