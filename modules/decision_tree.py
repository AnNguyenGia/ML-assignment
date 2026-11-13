from sklearn.tree import DecisionTreeClassifier


def train_decision_tree(X_train, y_train, X_test):
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    return model, model.predict(X_test)