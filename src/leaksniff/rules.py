"""The checks. One pass collects facts about the script, then each rule reads those facts.

Every rule is static and conservative: it would rather miss a leak than cry wolf on clean code.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field

from .core import Finding

SUPERVISED = {"SelectKBest", "SelectPercentile", "SelectFromModel", "RFE", "RFECV", "TargetEncoder",
              "SequentialFeatureSelector"}
TRANSFORMERS = {
    "StandardScaler", "MinMaxScaler", "RobustScaler", "MaxAbsScaler", "Normalizer", "QuantileTransformer",
    "PowerTransformer", "SimpleImputer", "KNNImputer", "IterativeImputer", "PCA", "TruncatedSVD",
    "KernelPCA", "FastICA", "NMF", "SelectKBest", "SelectPercentile", "SelectFromModel", "RFE", "RFECV",
    "VarianceThreshold", "SequentialFeatureSelector", "TargetEncoder",
    "PolynomialFeatures", "KBinsDiscretizer", "TfidfVectorizer", "CountVectorizer",
}
SPLITTERS = {"KFold", "StratifiedKFold", "GroupKFold", "StratifiedGroupKFold", "ShuffleSplit",
             "StratifiedShuffleSplit", "GroupShuffleSplit", "TimeSeriesSplit", "RepeatedKFold",
             "RepeatedStratifiedKFold", "LeaveOneGroupOut", "LeavePGroupsOut", "LeaveOneOut",
             "PredefinedSplit"}
RANDOM_SPLITTERS = {"KFold", "StratifiedKFold", "ShuffleSplit", "StratifiedShuffleSplit",
                    "RepeatedKFold", "RepeatedStratifiedKFold"}
CV_FUNCS = {"cross_val_score", "cross_validate", "cross_val_predict", "learning_curve",
            "validation_curve", "permutation_test_score"}
SEARCHES = {"GridSearchCV", "RandomizedSearchCV", "HalvingGridSearchCV", "HalvingRandomSearchCV"}
OTHER_METRICS = {"f1_score", "recall_score", "precision_score", "roc_auc_score", "average_precision_score",
                 "balanced_accuracy_score", "classification_report", "matthews_corrcoef", "fbeta_score",
                 "confusion_matrix", "precision_recall_curve", "precision_recall_fscore_support"}
IMBALANCE_HINTS = {"SMOTE", "ADASYN", "BorderlineSMOTE", "RandomOverSampler", "RandomUnderSampler",
                   "SMOTENC", "SMOTEENN", "SMOTETomek"}
EVAL_METHODS = {"predict", "predict_proba", "decision_function", "score", "evaluate"}
ID_COLUMN = re.compile(r"^(?:patient|customer|user|subject|participant|player|account|member|client|"
                       r"household|session|device|student|person|driver|employee|visitor|"
                       r"cust|pat|usr)[_ ]?(?:id|no|num|number|key|uuid)?$", re.I)


def name_of(node: ast.AST) -> str:
    """Last dotted part of a name: sklearn.preprocessing.StandardScaler -> StandardScaler."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _has(node: ast.AST, word: str, other: str) -> bool:
    """A name like X_test counts; train_test_size mentions both and counts as neither."""
    return any(isinstance(n, ast.Name) and word in n.id.lower() and other not in n.id.lower()
               for n in ast.walk(node))


def is_testy(node: ast.AST) -> bool:
    return _has(node, "test", "train")


def is_trainy(node: ast.AST) -> bool:
    return _has(node, "train", "test")


def kwarg(call: ast.Call, key: str):
    for k in call.keywords:
        if k.arg == key:
            return k.value
    return None


def constant(node) -> object:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant):
        return -node.operand.value
    return None


@dataclass
class Facts:
    calls: list[ast.Call] = field(default_factory=list)
    assigned_from: dict[str, str] = field(default_factory=dict)   # variable -> constructor name
    loops: list[ast.AST] = field(default_factory=list)            # for/while loops that are not CV loops
    strings: set[str] = field(default_factory=set)
    columns: set[str] = field(default_factory=set)                # strings used as df["col"] or column lists
    names: set[str] = field(default_factory=set)
    targets: set[str] = field(default_factory=set)                # column names used as y
    split_lines: list[int] = field(default_factory=list)          # where data gets split / cross-validated
    random_split_lines: list[int] = field(default_factory=list)   # splits that ignore time and groups
    group_aware: bool = False
    time_aware: bool = False
    temporal_data: bool = False


def _collect(tree: ast.AST) -> Facts:
    f = Facts()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f.calls.append(node)
        elif isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            ctor = name_of(node.value.func)
            for t in node.targets:
                if isinstance(t, ast.Name):
                    f.assigned_from[t.id] = ctor
        elif isinstance(node, (ast.For, ast.While)) and not _is_cv_loop(node):
            f.loops.append(node)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            f.strings.add(node.value)
        if isinstance(node, ast.Subscript):
            f.columns |= {n.value for n in ast.walk(node.slice) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
        if isinstance(node, ast.Call) and name_of(node.func) in {"drop", "groupby", "set_index", "read_csv"}:
            for part in list(node.args) + [k.value for k in node.keywords]:
                f.columns |= {n.value for n in ast.walk(part) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
        elif isinstance(node, ast.Name):
            f.names.add(node.id)
        elif isinstance(node, ast.alias):
            f.names.add(node.name.split(".")[-1])
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Subscript):
            if any(isinstance(t, ast.Name) and t.id.lower() in {"y", "target", "label", "labels"}
                   for t in node.targets):
                key = node.value.slice
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    f.targets.add(key.value)
    f.calls.sort(key=lambda c: (c.lineno, c.col_offset))
    _collect_splits(f)
    return f


def _is_cv_loop(node: ast.AST) -> bool:
    """`for train_idx, test_idx in kf.split(X)` is cross-validation, not test-set peeking."""
    if isinstance(node, ast.For):
        if is_testy(node.target):
            return True
        return any(isinstance(n, ast.Attribute) and n.attr == "split" for n in ast.walk(node.iter))
    return False


def _ctor(f: Facts, node: ast.AST) -> str:
    """Constructor behind an expression: `KFold(5)` or a variable assigned from `KFold(5)`."""
    if isinstance(node, ast.Call):
        return name_of(node.func)
    if isinstance(node, ast.Name):
        return f.assigned_from.get(node.id, "")
    return ""


def _collect_splits(f: Facts) -> None:
    f.temporal_data = any(kwarg(c, "parse_dates") is not None or name_of(c.func) == "to_datetime"
                          for c in f.calls)
    f.group_aware = any("Group" in n for n in f.names if n in SPLITTERS) or any(
        kwarg(c, "groups") is not None for c in f.calls)
    f.time_aware = "TimeSeriesSplit" in f.names
    for c in f.calls:
        fn = name_of(c.func)
        if fn == "train_test_split":
            f.split_lines.append(c.lineno)
            shuffle = kwarg(c, "shuffle")
            if not (isinstance(shuffle, ast.Constant) and shuffle.value is False):
                f.random_split_lines.append(c.lineno)
        elif fn == "split" and isinstance(c.func, ast.Attribute) and _ctor(f, c.func.value) in SPLITTERS:
            f.split_lines.append(c.lineno)
            if _ctor(f, c.func.value) in RANDOM_SPLITTERS:
                f.random_split_lines.append(c.lineno)
        elif fn in CV_FUNCS or fn in SEARCHES or (fn == "fit" and _ctor(f, c.func.value) in SEARCHES):
            f.split_lines.append(c.lineno)
            cv = kwarg(c, "cv")
            if fn in CV_FUNCS or fn in SEARCHES:
                if cv is None or isinstance(constant(cv), int) or _ctor(f, cv) in RANDOM_SPLITTERS:
                    f.random_split_lines.append(c.lineno)


def _fit_data(call: ast.Call):
    return call.args[0] if call.args else kwarg(call, "X")


# ---------------------------------------------------------------- rules

def rule_preprocess_before_split(f: Facts):
    for c in f.calls:
        fn = name_of(c.func)
        if fn not in {"fit", "fit_transform", "fit_resample"} or not isinstance(c.func, ast.Attribute):
            continue
        data = _fit_data(c)
        if data is None:
            continue
        if is_testy(data):
            yield Finding("LS005", "error", c.lineno, f"`{fn}` is called on test data",
                          "Fit only on training data, then call transform/predict on the test set.")
            continue
        ctor = _ctor(f, c.func.value)
        if is_trainy(data) or not any(line > c.lineno for line in f.split_lines):
            continue
        if fn == "fit_resample":
            yield Finding("LS002", "error", c.lineno,
                          f"{ctor or 'resampling'} runs before the data is split or cross-validated",
                          "Synthetic rows from the test fold leak into training. Put the sampler inside an "
                          "imblearn Pipeline so it only runs on training folds.")
        elif ctor in SUPERVISED:
            yield Finding("LS001", "error", c.lineno,
                          f"{ctor} uses the labels of the full dataset before the split / cross-validation",
                          "It picks features using the answers of rows you later test on, which can make "
                          "random noise look predictive. Put it in a Pipeline so each fold fits its own copy.")
        elif ctor in TRANSFORMERS:
            yield Finding("LS001", "warning", c.lineno,
                          f"{ctor} is fitted on the full dataset before the split / cross-validation",
                          "It learns statistics from rows you later test on. Split first, or put it in a "
                          "Pipeline so each fold fits its own copy.")


def rule_time_order(f: Facts):
    if not f.temporal_data or f.time_aware:
        return
    for line in f.random_split_lines:
        yield Finding("LS003", "error", line, "random / shuffled split on time-ordered data",
                      "Each fold trains on days after the ones it tests on. Use TimeSeriesSplit, or "
                      "train_test_split(..., shuffle=False) after sorting by date.")


def rule_groups(f: Facts):
    if f.group_aware or f.temporal_data or not f.random_split_lines:
        return
    ids = sorted(s for s in f.columns if ID_COLUMN.search(s))
    if ids:
        yield Finding("LS004", "warning", f.random_split_lines[0],
                      f"rows look grouped by `{ids[0]}` but the split ignores groups",
                      "The same entity can land in train and test, so the model memorises it. Use "
                      "GroupKFold / GroupShuffleSplit with groups=.")


def _is_test_eval(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    fn = name_of(node.func)
    evaluates = fn in EVAL_METHODS or fn.endswith(("_score", "_error")) or fn in OTHER_METRICS
    return evaluates and any(is_testy(a) for a in node.args)


def _selects(loop: ast.AST) -> bool:
    """The loop keeps a winner: it compares a test-set score, or stores a best* variable."""
    scores = set()
    for n in ast.walk(loop):
        if isinstance(n, ast.Assign) and any(_is_test_eval(c) and _is_scalar_metric(c) for c in ast.walk(n.value)):
            scores |= {t.id for t in n.targets if isinstance(t, ast.Name)}
    for n in ast.walk(loop):
        if isinstance(n, ast.Name) and n.id.lower().startswith("best") and isinstance(n.ctx, ast.Store):
            return True
        if isinstance(n, (ast.If, ast.IfExp)) and any(
                isinstance(x, ast.Name) and x.id in scores for x in ast.walk(n.test)):
            return True
    return False


def _is_scalar_metric(call: ast.Call) -> bool:
    fn = name_of(call.func)
    return fn in {"score", "evaluate"} or fn.endswith(("_score", "_error", "_loss")) or fn == "log_loss"


def rule_test_reuse(f: Facts):
    for loop in f.loops:
        if not _selects(loop):
            continue
        for node in ast.walk(loop):
            if not isinstance(node, ast.Call):
                continue
            if _is_test_eval(node):
                yield Finding("LS005", "warning", node.lineno,
                              "the test set is scored inside a loop, so it is being used to pick a model",
                              "Choose models with cross-validation on training data and score the test "
                              "set once at the end.")


def rule_future_features(f: Facts):
    for c in f.calls:
        fn = name_of(c.func)
        if fn == "shift":
            n = constant(c.args[0]) if c.args else constant(kwarg(c, "periods"))
            if isinstance(n, (int, float)) and n < 0:
                yield Finding("LS006", "error", c.lineno, f"shift({n}) pulls values from the future",
                              "A negative shift copies later rows into the current one. Use positive lags.")
        elif fn == "rolling" and constant(kwarg(c, "center")) is True:
            yield Finding("LS006", "error", c.lineno, "centred rolling window uses future rows",
                          "Drop center=True and shift(1) first so the window only sees the past.")


def rule_target_encoding(f: Facts):
    first_split = min(f.split_lines, default=None)
    for c in f.calls:
        if name_of(c.func) != "transform" or not c.args or constant(c.args[0]) not in {"mean", "median"}:
            continue
        sub = c.func.value if isinstance(c.func, ast.Attribute) else None
        if not isinstance(sub, ast.Subscript) or not isinstance(sub.slice, ast.Constant):
            continue
        if sub.slice.value in f.targets and (first_split is None or c.lineno < first_split):
            yield Finding("LS007", "error", c.lineno,
                          f"target `{sub.slice.value}` is averaged over all rows to build a feature",
                          "Each row's feature includes its own label. Compute the encoding on training "
                          "folds only (sklearn TargetEncoder inside a Pipeline does this).")


def rule_accuracy_only(f: Facts):
    imbalanced = bool(IMBALANCE_HINTS & f.names) or "balanced" in f.strings or any(
        kwarg(c, "class_weight") is not None for c in f.calls)
    other = (OTHER_METRICS & f.names) or any(
        s in f.strings for s in ("f1", "roc_auc", "average_precision", "balanced_accuracy", "recall", "precision"))
    if not imbalanced or other:
        return
    for c in f.calls:
        scoring = kwarg(c, "scoring")
        if name_of(c.func) == "accuracy_score" or constant(scoring) == "accuracy":
            yield Finding("LS008", "warning", c.lineno, "accuracy is the only metric on imbalanced data",
                          "Predicting the majority class already scores high. Report recall/precision, "
                          "F1, PR-AUC or balanced accuracy, and the majority-class baseline.")
            return


def rule_baseline(f: Facts):
    if any(n.startswith("Dummy") for n in f.names) or any("baseline" in n.lower() for n in f.names):
        return
    fits = [c for c in f.calls if name_of(c.func) == "fit"]
    if fits:
        yield Finding("LS009", "info", fits[0].lineno, "no baseline model to compare against",
                      "Score a DummyClassifier/DummyRegressor too, so a reader can tell whether the "
                      "model beats guessing.")


RULES = [rule_preprocess_before_split, rule_time_order, rule_groups, rule_test_reuse,
         rule_future_features, rule_target_encoding, rule_accuracy_only, rule_baseline]


def analyse(tree: ast.AST) -> list[Finding]:
    facts = _collect(tree)
    seen, out = set(), []
    for rule in RULES:
        for finding in rule(facts):
            key = (finding.code, finding.line)
            if key not in seen:
                seen.add(key)
                out.append(finding)
    return out
