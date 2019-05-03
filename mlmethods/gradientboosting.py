from mlmethods.basetrainers import BaseTrainer
from mlmethods.baseclassifiers import BaseClassifier
import mlmethods.constatns as const
import xgboost as xgb

class DefaultGradientBoostingTrainer(BaseTrainer):
    def get_trained_classifier(self, X_train, y_train):
        return self.__get_trained_simple_gradient_boosting(X_train, y_train)

    # this method for default training functionality 
    def __get_trained_simple_gradient_boosting(self, X_train, y_train):
        params = {
        const.XGBOOST_HYPER_PARAM_OBJECTIVE : const.XGBOOST_OBJECTIVE_BINARY_LOGISTIC,
        const.MAX_DEPTH : const.MAX_DEPTH_VALUE,
        const.XGBOOST_HYPER_PARAM_SILENT : const.XGBOOST_SILENT_VALUE,
        const.XGBOOST_HYPER_PARAM_ETA : const.XGBOOST_ETA_VALUE
    }
        classifier = DefaultGradientBoostingClassifier(params)
        classifier.train(X_train, y_train)
        return classifier

class DefaultGradientBoostingClassifier(BaseClassifier):
    def __init__(self, params, booster=None):
        self.params = params
        self.child_classifier = booster

    def train(self, X_train, y_train):
        self.child_classifier = xgb.train(self.params, xgb.DMatrix(X_train, label=y_train), num_boost_round=const.XGBOOST_NUM_BOOST_ROUND) 

    def predict(self, X_holdout):
        return self.child_classifier.predict(xgb.DMatrix(X_holdout))

        