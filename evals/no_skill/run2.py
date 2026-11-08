import numpy as np
import pandas as pd
df = pd.read_csv("daily_sales.csv", parse_dates=["date"])
g = df.groupby("store_id")["sales"]
df["lag_1"] = g.shift(1)
df["roll_mean_7"] = g.transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = []
for tr, te in kf.split(df):
    tr_df, te_df = df.iloc[tr], df.iloc[te]
    model.fit(tr_df[FEATURES], tr_df["sales"])
    scores.append(r2_score(te_df["sales"], model.predict(te_df[FEATURES])))
