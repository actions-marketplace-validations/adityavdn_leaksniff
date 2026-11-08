import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, average_precision_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

NUM = ["tenure", "monthly_spend", "support_tickets"]
CAT = ["plan", "region"]
df = pd.read_csv("churn_monthly.csv")
X = df[NUM + CAT]
y = df["churned"]
groups = df["customer_id"]
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups))
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
groups_train = groups.iloc[train_idx]
preprocess = ColumnTransformer([
    ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), NUM),
    ("cat", Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore"))]), CAT),
])
pipe = ImbPipeline([("prep", preprocess), ("smote", SMOTE(random_state=42)),
                    ("rf", RandomForestClassifier(random_state=42))])
search = RandomizedSearchCV(pipe, {"rf__min_samples_leaf": [1, 5, 20]}, n_iter=3,
                            scoring="average_precision", cv=GroupKFold(n_splits=3), random_state=42)
search.fit(X_train, y_train, groups=groups_train)
best = search.best_estimator_
y_pred = best.predict(X_test)
y_prob = best.predict_proba(X_test)[:, 1]
baseline_acc = 1 - y_test.mean()
print(accuracy_score(y_test, y_pred), baseline_acc, recall_score(y_test, y_pred))
print(average_precision_score(y_test, y_prob))
