from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=SEED)
clf.fit(train_X, train_y)
print(f"Train accuracy: {clf.score(train_X, train_y):.3f}")
