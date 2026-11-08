from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score

def make_search(clf, grid):
    pipe = Pipeline([("select", SelectKBest(f_classif, k=50)), ("scale", StandardScaler()), ("clf", clf)])
    return GridSearchCV(pipe, grid, scoring="roc_auc", cv=StratifiedKFold(5, shuffle=True, random_state=0))

search = make_search(LogisticRegression(max_iter=5000), {"clf__C": [0.01, 0.1, 1]})
scores = cross_val_score(search, X_train, y_train, scoring="roc_auc",
                         cv=StratifiedKFold(5, shuffle=True, random_state=1))
best = search.fit(X_train, y_train)
test_auc = roc_auc_score(y_test, best.predict_proba(X_test)[:, 1])
print(scores.mean(), test_auc)
