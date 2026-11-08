import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit

df = pd.read_csv("daily_sales.csv", parse_dates=["date"]).sort_values(["store_id", "date"])
g = df.groupby("store_id")["sales"]
for lag in (1, 7, 14, 28):
    df[f"lag_{lag}"] = g.shift(lag)
for w in (7, 28):
    df[f"roll_mean_{w}"] = g.transform(lambda s: s.shift(1).rolling(w).mean())
df["dow"] = df["date"].dt.dayofweek
df = df.dropna().reset_index(drop=True)
FEATS = ["promo", "temperature", "dow", "store_te", "lag_1", "lag_7", "roll_mean_7"]
dates = np.sort(df["date"].unique())
scores = []
for fold, (tr_i, te_i) in enumerate(TimeSeriesSplit(n_splits=5).split(dates), 1):
    tr = df[df["date"].isin(dates[tr_i])].copy()
    te = df[df["date"].isin(dates[te_i])].copy()
    store_mean = tr.groupby("store_id")["sales"].mean()
    tr["store_te"] = tr["store_id"].map(store_mean)
    te["store_te"] = te["store_id"].map(store_mean).fillna(tr["sales"].mean())
    model = LGBMRegressor(n_estimators=500, random_state=42)
    model.fit(tr[FEATS], tr["sales"])
    scores.append(r2_score(te["sales"], model.predict(te[FEATS])))
print(np.mean(scores))
