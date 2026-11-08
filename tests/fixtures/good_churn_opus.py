import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, average_precision_score, recall_score, precision_score
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUM_COLS = ["tenure", "monthly_spend", "support_tickets"]
CAT_COLS = ["plan", "region"]
df = pd.read_csv("churn_monthly.csv")
df["month"] = pd.to_datetime(df["month"], errors="coerce")
df = df.dropna(subset=["month"]).sort_values(["month", "customer_id"])
months = np.sort(df["month"].unique())
cutoff = months[int(len(months) * 0.8)]
train, test = df[df["month"] < cutoff], df[df["month"] >= cutoff]
X_train, y_train, g_train = train[NUM_COLS + CAT_COLS], train["churned"], train["customer_id"]
X_test, y_test = test[NUM_COLS + CAT_COLS], test["churned"]
preprocess = ColumnTransformer([
    ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), NUM_COLS),
    ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                      ("onehot", OneHotEncoder(handle_unknown="ignore"))]), CAT_COLS),
])
pipe = Pipeline([("prep", preprocess), ("smote", SMOTE(random_state=42)),
                 ("rf", RandomForestClassifier(random_state=42))])
search = RandomizedSearchCV(pipe, {"rf__max_depth": [None, 8, 12]}, n_iter=3,
                            scoring="average_precision", cv=GroupKFold(n_splits=5), random_state=42)
search.fit(X_train, y_train, groups=g_train)
model = search.best_estimator_
proba = model.predict_proba(X_test)[:, 1]
pred = (proba >= 0.5).astype(int)
baseline = 1 - y_test.mean()
print(accuracy_score(y_test, pred), baseline, recall_score(y_test, pred), precision_score(y_test, pred))
print(average_precision_score(y_test, proba))
