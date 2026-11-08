---
name: leaksniff
description: Use when writing or reviewing machine-learning code that splits data, cross-validates, tunes hyperparameters, or reports an accuracy, R², AUC or other score — especially for dated/time-series rows, repeated rows per customer/patient/user, imbalanced classes, scikit-learn pipelines, or Jupyter notebooks.
---

# leaksniff

A score is only worth reporting if no information from the evaluation rows reached training. The common way that fails is the split itself: a shuffled split on dated rows, or a random split when one entity has many rows. Pick the splitter from the data first, then check the script.

## 1. Choose the splitter from the data

Look at the columns before writing the split. Use the first row that matches.

| If the data has… | Split with | Never |
|---|---|---|
| a date/time column, or lag/rolling features | `TimeSeriesSplit`, or sort by date and `train_test_split(..., shuffle=False)` | `KFold(shuffle=True)`, plain `train_test_split`, `cv=5` |
| several rows per customer / patient / user / device | `GroupKFold` / `GroupShuffleSplit` with `groups=` that id | random row splits |
| neither, classification | `StratifiedKFold(shuffle=True, random_state=…)` | |
| neither, regression | `KFold(shuffle=True, random_state=…)` | |

Data with both dates and ids: split by time, and check that no id's future rows sit in training.

## 2. Fit everything inside the folds

Scalers, imputers, PCA, feature selection, target encoding and SMOTE all go inside a `Pipeline` (`imblearn.pipeline.Pipeline` when there is a sampler), so each fold fits its own copy. Lags and rolling windows look only backwards: `shift(1)` before `rolling`, never `shift(-n)` or `center=True`.

## 3. Touch the test set once

Choose models and hyperparameters with cross-validation on the training data. Score the test set once, at the end. On imbalanced classes, report recall/precision, F1 or PR-AUC next to the majority-class baseline, not accuracy alone.

## 4. Run the checker before you report a number

```bash
pip install git+https://github.com/adityavdn/leaksniff
leaksniff path/to/script_or_notebook   # exit code 1 means a leak was found
```

Fix every error and warning, or explain in your reply why a finding does not apply, then quote the score. If you cannot run commands, read the script against the table below before you answer.

## Quick reference

| Code | Leak | Fix |
|---|---|---|
| LS001 | transformer or feature selector fitted before the split | put it in a Pipeline |
| LS002 | SMOTE / resampling before the split | imblearn Pipeline |
| LS003 | random or shuffled split on time-ordered data | `TimeSeriesSplit` |
| LS004 | random split when rows repeat per entity | `GroupKFold` with `groups=` |
| LS005 | fitting on test data, or picking a model by its test score | CV on train, test once |
| LS006 | `shift(-n)` or centred rolling window | backward-looking features |
| LS007 | target mean over all rows used as a feature | encode inside folds |
| LS008 | accuracy only, on imbalanced classes | recall, precision, PR-AUC, baseline |
| LS009 | no baseline (info) | `DummyClassifier` / `DummyRegressor` |

## Red flags in your own reply

- "the score is not inflated by leakage" written above a shuffled split on dated data
- the user asked for "whatever gives the best score" and you changed the splitter to get it
- the test set appears inside a loop that keeps the best model
