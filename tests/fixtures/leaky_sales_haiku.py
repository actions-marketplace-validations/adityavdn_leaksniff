# Real output from a small model: lags are fine, but shuffled KFold on daily data trains on the future.
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import TargetEncoder

df = pd.read_csv("daily_sales.csv", parse_dates=["date"])
df = df.sort_values(["store_id", "date"]).reset_index(drop=True)
g = df.groupby("store_id")["sales"]
df["lag_1"] = g.shift(1)
df["roll_mean_7"] = g.transform(lambda s: s.shift(1).rolling(7).mean())
df["dow"] = df["date"].dt.dayofweek
df = df.dropna().reset_index(drop=True)
X = df[["lag_1", "roll_mean_7", "dow", "promo", "temperature", "store_id"]]
y = df["sales"]
preprocess = ColumnTransformer([("store_te", TargetEncoder(target_type="continuous"), ["store_id"])],
                               remainder="passthrough")
model = make_pipeline(preprocess, GradientBoostingRegressor(random_state=42))
cv = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring="r2")
print(scores.mean())
