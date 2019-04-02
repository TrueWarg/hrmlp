class BaseTrainer:
    def get_trained_classifier(self, X_train, y_train):
        raise NotImplementedError("Subclass must implement get_trained_classifier method")