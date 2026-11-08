import numpy as np
import pandas as pd
df = pd.read_csv("daily_sales.csv", parse_dates=["date"])
g = df.groupby("store_id")["sales"]
df["lag_1"] = g.shift(1)
df["roll_mean_7"] = g.transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
df = df.sort_values(["date", "store_id"]).reset_index(drop=True)
scores = []
for tr, te in KFold(n_splits=5, shuffle=False).split(X_all):
    model.fit(X_all.iloc[tr], y_all.iloc[tr])
    scores.append(r2_score(y_all.iloc[te], model.predict(X_all.iloc[te])))
