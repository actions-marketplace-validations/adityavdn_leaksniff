import numpy as np
import pandas as pd
df = pd.read_csv("daily_sales.csv", parse_dates=["date"])
g = df.groupby("store_id")["sales"]
df["lag_1"] = g.shift(1)
df["roll_mean_7"] = g.transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())
from sklearn.dummy import DummyRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit
dates = np.sort(df["date"].unique())
model_scores, base_scores = [], []
for train_idx, test_idx in TimeSeriesSplit(n_splits=5).split(dates):
    tr = df[df["date"].isin(dates[train_idx])]
    te = df[df["date"].isin(dates[test_idx])]
    pipe.fit(tr[FEATURES], tr["sales"])
    model_scores.append(r2_score(te["sales"], pipe.predict(te[FEATURES])))
    base = DummyRegressor(strategy="mean").fit(tr[FEATURES], tr["sales"])
    base_scores.append(r2_score(te["sales"], base.predict(te[FEATURES])))
