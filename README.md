# leaksniff

**Your model's score is only as honest as its split.** leaksniff reads your Python scripts and Jupyter notebooks and points at the lines where test data leaks into training: a scaler fitted before the split, SMOTE run on the whole dataset, a shuffled K-fold on daily sales, a test set used to pick the best model.

It ships as two things:

- a **command-line checker** (pure Python, no dependencies, works on `.py` and `.ipynb`)
- an **agent skill** for Claude Code, Codex, Cursor and other coding agents, so the model writing your ML code picks the right split before you ever run it

```text
$ leaksniff churn.ipynb
churn.ipynb:cell 4:2: LS001 warning: StandardScaler is fitted on the full dataset before the split / cross-validation
    fix: It learns statistics from rows you later test on. Split first, or put it in a Pipeline so each fold fits its own copy.
churn.ipynb:cell 4:3: LS002 error: SMOTE runs before the data is split or cross-validated
    fix: Synthetic rows from the test fold leak into training. Put the sampler inside an imblearn Pipeline so it only runs on training folds.
churn.ipynb:cell 6:1: LS004 warning: rows look grouped by `customer_id` but the split ignores groups
    fix: The same entity can land in train and test, so the model memorises it. Use GroupKFold / GroupShuffleSplit with groups=.

3 problem(s) in 1 of 1 file(s).
```

## Why

Leakage is the most common reason a model that scored 0.95 in a notebook falls apart in production or in a marker's hands. It rarely throws an error. The number just looks great.

Coding agents make the same mistakes. In a small experiment (see [evals/](evals/)), a small model asked to cross-validate a daily sales forecast used shuffled K-fold, which trains on the future, in **4 out of 4** runs, and each time it wrote that the score was "not inflated by leakage". With the skill loaded it used `TimeSeriesSplit` in **3 out of 3** runs and added a baseline model unprompted.

## Install

```bash
pip install git+https://github.com/adityavdn/leaksniff
```

Python 3.10+. No other dependencies: leaksniff never imports or runs your code, it reads it.

## Use

```bash
leaksniff                      # check everything under the current folder
leaksniff train.py analysis.ipynb
leaksniff src/ --min-level info --ignore LS009
leaksniff . --format json      # for scripts and CI
```

The exit code is 1 when anything at warning level or above is found (change with `--fail-on error`). Silence a line you have checked with `# noqa: LS004`.

**Pre-commit**

```yaml
- repo: https://github.com/adityavdn/leaksniff
  rev: v0.1.0
  hooks:
    - id: leaksniff
```

**GitHub Actions**

```yaml
- uses: actions/checkout@v4
- uses: adityavdn/leaksniff@v0.1.0
  with:
    paths: notebooks/ src/
```

**Coding agents.** Copy [`skills/leaksniff`](skills/leaksniff) into your agent's skills folder (`~/.claude/skills/` for Claude Code, `~/.agents/skills/` for Codex and others). The agent will then choose the splitter from the data, keep preprocessing inside folds, and run the checker before it quotes a score.

## What it checks

| Code | Level | Catches |
|---|---|---|
| LS001 | error / warning | a scaler, imputer, PCA (warning) or feature selector / target encoder (error) fitted on all rows before the split or cross-validation |
| LS002 | error | SMOTE or another resampler run before the split |
| LS003 | error | a shuffled or plain K-fold / `train_test_split` on time-ordered data |
| LS004 | warning | a random split when rows repeat per patient / customer / user |
| LS005 | error / warning | fitting on test data, or a loop that keeps the model with the best test score |
| LS006 | error | `shift(-n)` or `rolling(center=True)`, which copy future values into today's row |
| LS007 | error | a group mean of the target, computed over all rows, used as a feature |
| LS008 | warning | accuracy as the only metric on imbalanced classes |
| LS009 | info | no baseline model (`DummyClassifier` / `DummyRegressor`) to compare against |

Everything inside a scikit-learn or imblearn `Pipeline` is fine, because each fold fits its own copy. Fitting on `X_train` after the split is fine. Comparing several models on the test set for a report is fine; picking the winner by that score is not.

## How well it works

leaksniff is a static checker, so it reads patterns and cannot be certain about intent. It is tuned to stay quiet on correct code:

- **scikit-learn's own 282 example scripts:** 7 findings, all genuine fits before a split. One is `svm/plot_rbf_parameters.py`, which scales the whole dataset before `GridSearchCV` (the example's own comments admit to "cheating a bit"). The rest are scalers or PCA fitted on all rows for illustration.
- **Clean code written by agents:** the four leak-free scripts in [`tests/fixtures`](tests/fixtures) produce no warnings.
- **Six student ML notebooks:** one finding, a `MinMaxScaler` fitted before `train_test_split`.

If it flags something that is not a leak, please [open an issue](https://github.com/adityavdn/leaksniff/issues) with the snippet. False positives are bugs.

## Limits

It does not follow data through functions you wrote yourself, it cannot see leaks hidden in SQL or upstream feature stores, and it judges "time-ordered" from `parse_dates` / `to_datetime` in the same file. A clean run means none of these patterns were found, not that the evaluation is proven correct.

## Development

```bash
pip install -e . pytest
pytest
```

Every rule has a test that must fire and a test that must stay quiet. A new rule needs both.

## License

MIT
