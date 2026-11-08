import numpy as np
import pandas as pd
df = pd.read_csv("daily_sales.csv", parse_dates=["date"])
g = df.groupby("store_id")["sales"]
df["lag_1"] = g.shift(1)
df["roll_mean_7"] = g.transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())
from sklearn.model_selection import KFold, cross_val_score
cv = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring="r2", n_jobs=-1)
