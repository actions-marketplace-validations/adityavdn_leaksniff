# Every classic mistake in one script. leaksniff must flag each one.
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE

df = pd.read_csv("patients.csv")
df["dept_mean"] = df.groupby("department")["readmitted"].transform("mean")   # LS007
df["next_visit"] = df["visits"].shift(-1)                                    # LS006
df["smooth"] = df["visits"].rolling(7, center=True).mean()                   # LS006
X = df.drop(columns=["readmitted", "patient_id"])
y = df["readmitted"]

X = StandardScaler().fit_transform(X)                                        # LS001
X = SelectKBest(k=20).fit_transform(X, y)                                    # LS001
X, y = SMOTE().fit_resample(X, y)                                            # LS002

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)     # LS004 (patient_id)

best = None
for depth in [3, 5, 10, None]:
    m = RandomForestClassifier(max_depth=depth).fit(X_train, y_train)
    acc = accuracy_score(y_test, m.predict(X_test))                          # LS005, LS008
    if best is None or acc > best[0]:
        best = (acc, m)
print(best[0])
