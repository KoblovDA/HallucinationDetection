from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

clf = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced",
                              solver="lbfgs", random_state=SEED)),
])
clf.fit(train_X, train_y)
train_acc = clf.score(train_X, train_y)
print(f"Train accuracy: {train_acc:.3f}")
