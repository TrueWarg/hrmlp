from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from mlmethods.basetrainers import BaseTrainer
from mlmethods.baseclassifiers import BaseClassifier
import mlmethods.constatns as const
import numpy as np

class DefaultLogisticRegressionTrainer(BaseTrainer):
    def get_trained_classifier(self, X_train, y_train):
        return self.__get_trained_logistic_regression_search_cv(X_train, y_train)

    # this method for default training functionality 
    def __get_trained_logistic_regression_search_cv(self, X_train, y_train):
        skf = StratifiedKFold(n_splits=const.CROSS_VALIDATION_COUNT)
        c_values = c_values = np.logspace(
            const.REGRESSION_COEF_SELECTION_POWER_BOOTOM_VALUE, 
            const.REGRESSION_COEF_SELECTION_POWER_TOP_VALUE,  
            const.REGRESSION_COEF_SELECTION_VALUES_COUNT
        )
        logistic_regression_grid = LogisticRegressionCV(Cs=c_values, cv=skf, n_jobs=const.JOBS_NUMBER)

        classifier = DefaultLogisticRegressionClassifier(logistic_regression_grid)
        classifier.train(X_train, y_train)
        return classifier

class DefaultLogisticRegressionClassifier(BaseClassifier):
    def __init__(self, logistic_regression_grid):
        self.child_classifier = logistic_regression_grid

    def train(self, X_train, y_train):
        self.child_classifier.fit(X_train, y_train)

    def predict(self, X_holdout):
        return self.child_classifier.predict(X_holdout)
