import json
from pathlib import Path

import pytest

from leaksniff import check_source, check_path
from leaksniff.cli import main

FIX = Path(__file__).parent / "fixtures"


def codes(findings, min_level="warning"):
    order = {"info": 0, "warning": 1, "error": 2}
    return sorted({f.code for f in findings if order[f.level] >= order[min_level]})


def lines_for(findings, code):
    return sorted(f.line for f in findings if f.code == code)


# ---------- the textbook leaky script: every rule fires on the right line ----------

def test_classic_script_flags_every_mistake():
    found = check_path(FIX / "leaky_classic.py")
    assert codes(found) == ["LS001", "LS002", "LS004", "LS005", "LS006", "LS007", "LS008"]
    assert lines_for(found, "LS001") == [17, 18]
    assert lines_for(found, "LS002") == [19]
    assert lines_for(found, "LS006") == [12, 13]
    assert lines_for(found, "LS007") == [11]
    assert lines_for(found, "LS005") == [26]


def test_real_small_model_output_is_caught():
    found = check_path(FIX / "leaky_sales_haiku.py")
    assert codes(found) == ["LS003"]


# ---------- clean scripts written by agents must produce zero warnings ----------

@pytest.mark.parametrize("name", ["good_churn_opus.py", "good_churn_haiku.py",
                                  "good_sales_opus.py", "good_genes_opus.py"])
def test_clean_scripts_have_no_warnings(name):
    assert codes(check_path(FIX / name)) == []


# ---------- individual rules, positive and negative ----------

def test_fit_inside_pipeline_is_fine():
    src = """
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
pipe = make_pipeline(StandardScaler(), LogisticRegression()).fit(X_train, y_train)
"""
    assert codes(check_source(src)) == []


def test_scaler_fit_after_split_on_train_is_fine():
    src = """
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
"""
    assert codes(check_source(src)) == []


def test_scaler_fit_before_cross_val_score_is_flagged():
    src = """
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
print(cross_val_score(model, X, y, cv=5))
"""
    assert codes(check_source(src)) == ["LS001"]


def test_fit_on_test_data_is_flagged():
    src = """
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_test = sc.fit_transform(X_test)
"""
    assert "LS005" in codes(check_source(src))


def test_time_series_split_is_fine():
    src = """
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
df = pd.read_csv("x.csv", parse_dates=["date"])
print(cross_val_score(model, X, y, cv=TimeSeriesSplit(5)))
"""
    assert codes(check_source(src)) == []


def test_train_test_split_on_dated_data_is_flagged_unless_unshuffled():
    leaky = """
import pandas as pd
from sklearn.model_selection import train_test_split
df = pd.read_csv("x.csv")
df["date"] = pd.to_datetime(df["date"])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
"""
    ok = leaky.replace("test_size=0.2", "test_size=0.2, shuffle=False")
    assert codes(check_source(leaky)) == ["LS003"]
    assert codes(check_source(ok)) == []


def test_group_kfold_silences_group_rule():
    src = """
from sklearn.model_selection import GroupKFold, cross_val_score
groups = df["patient_id"]
print(cross_val_score(model, X, y, groups=groups, cv=GroupKFold(5)))
"""
    assert codes(check_source(src)) == []


def test_accuracy_with_other_metrics_is_fine():
    src = """
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, f1_score
print(accuracy_score(y_test, p), f1_score(y_test, p))
"""
    assert "LS008" not in codes(check_source(src))


def test_shift_positive_is_fine():
    assert codes(check_source("df['lag'] = df['x'].shift(1)\n")) == []


def test_missing_baseline_is_info_only():
    src = """
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)
RandomForestClassifier().fit(X_train, y_train)
"""
    found = check_source(src)
    assert codes(found) == []
    assert codes(found, "info") == ["LS009"]


def test_noqa_suppresses_a_line():
    src = "df['next'] = df['x'].shift(-1)  # noqa: LS006\n"
    assert codes(check_source(src)) == []


def test_syntax_error_is_reported_not_raised():
    found = check_source("def broken(:\n")
    assert codes(found, "info") == ["LS000"]


# ---------- notebooks and CLI ----------

def test_notebook_cells_are_checked_with_cell_numbers(tmp_path):
    nb = {"cells": [
        {"cell_type": "markdown", "source": ["# title"]},
        {"cell_type": "code", "source": ["from sklearn.preprocessing import StandardScaler\n",
                                         "from sklearn.model_selection import train_test_split\n"]},
        {"cell_type": "code", "source": ["%matplotlib inline\n", "X = StandardScaler().fit_transform(X)\n",
                                         "a, b, c, d = train_test_split(X, y)\n"]},
    ], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}
    path = tmp_path / "work.ipynb"
    path.write_text(json.dumps(nb))
    found = check_path(path)
    hit = [f for f in found if f.code == "LS001"][0]
    assert hit.cell == 3 and hit.cell_line == 2


def test_cli_exit_codes_and_json(tmp_path, capsys):
    assert main([str(FIX / "good_genes_opus.py")]) == 0
    assert main([str(FIX / "leaky_classic.py"), "--format", "json"]) == 1
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert {f["code"] for f in out} >= {"LS001", "LS002"}


def test_cli_scans_directories(capsys):
    assert main([str(FIX)]) == 1
    text = capsys.readouterr().out
    assert "leaky_classic.py" in text and "good_genes_opus.py" not in text


# ---------- regressions found by running on scikit-learn's own examples ----------

def test_name_with_train_and_test_is_not_test_data():
    src = "for train_test_size in sizes:\n    est.fit(X[:train_test_size], y[:train_test_size])\n"
    assert codes(check_source(src)) == []


def test_comparing_models_on_test_for_a_report_is_fine():
    src = """
for name, model in models.items():
    model.fit(X_train, y_train)
    print(name, model.score(X_test, y_test))
"""
    assert codes(check_source(src)) == []


def test_unsupervised_fit_is_warning_supervised_is_error():
    src = """
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import cross_val_score
Xp = PCA(10).fit_transform(X)
Xs = SelectKBest(k=10).fit_transform(X, y)
cross_val_score(model, Xs, y, cv=StratifiedKFold(5))
"""
    levels = {f.code + str(f.line): f.level for f in check_source(src)}
    assert levels["LS0015"] == "warning" and levels["LS0016"] == "error"


def test_id_string_that_is_not_a_column_is_ignored():
    src = """
from sklearn.model_selection import train_test_split
tracker = Monitor(name="eval_task_id")
X_train, X_test, y_train, y_test = train_test_split(X, y)
"""
    assert codes(check_source(src)) == []


def test_unrelated_if_in_comparison_loop_is_fine():
    src = """
for name, clf in models:
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    if name == "svm":
        print("kernel model")
"""
    assert codes(check_source(src)) == []


def test_keeping_the_best_test_score_is_flagged():
    src = """
best_auc = 0
for C in [0.1, 1, 10]:
    m = LogisticRegression(C=C).fit(X_train, y_train)
    auc = roc_auc_score(y_test, m.predict_proba(X_test)[:, 1])
    if auc > best_auc:
        best_auc = auc
"""
    assert codes(check_source(src)) == ["LS005"]


def test_corrupt_notebook_is_reported_not_raised(tmp_path):
    path = tmp_path / "broken.ipynb"
    path.write_text("\n<<<<<<< HEAD\n")
    assert codes(check_path(path), "info") == ["LS000"]


def test_category_encoders_before_split_are_fine():
    src = """
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
df["city"] = LabelEncoder().fit_transform(df["city"])
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)
"""
    assert codes(check_source(src)) == []


# ---------- the agent experiment in evals/ (see README) ----------

EVALS = Path(__file__).parent.parent / "evals"


@pytest.mark.parametrize("path", sorted((EVALS / "no_skill").glob("*.py")), ids=lambda p: p.name)
def test_agent_runs_without_skill_leak(path):
    assert codes(check_path(path)) == ["LS003"]


@pytest.mark.parametrize("path", sorted((EVALS / "with_skill").glob("*.py")), ids=lambda p: p.name)
def test_agent_runs_with_skill_are_clean(path):
    assert codes(check_path(path), "info") == []
