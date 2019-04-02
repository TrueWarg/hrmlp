class BaseClassifier():
    def train(self, X_train, y_train):
        raise NotImplementedError("Subclass must implement train method")

    def predict(self, X_holdout):
        raise NotImplementedError("Subclass must implement predict method")