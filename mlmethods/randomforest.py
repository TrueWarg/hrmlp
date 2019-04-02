from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from mlmethods.basetrainers import BaseTrainer
from mlmethods.baseclassifiers import BaseClassifier
import mlmethods.constatns as const

class DefaultRandomForestTrainer(BaseTrainer):
    def get_trained_classifier(self, X_train, y_train):
        return self.__train_simple_random_forest(X_train, y_train)

    # this method for default training functionality 
    def __train_simple_random_forest(self, X_train, y_train):
        forest = RandomForestClassifier(
            n_estimators=const.RANDOM_FOREST_ESTIMATORS_COUNT, 
            max_depth=const.RANDOM_FORES_TOP_MAX_DEPTH_VALUE,
            n_jobs=const.JOBS_NUMBER,
            oob_score=const.RANDOM_FOREST_OOB_SCORE
        )
        classifier = DefaultKNeighborsClassifier(forest)
        classifier.train(X_train, y_train)
        return classifier

class DefaultKNeighborsClassifier(BaseClassifier):

    def __init__(self, forest):
        self.forest = forest

    def train(self, X_train, y_train):
        self.forest.fit(X_train, y_train)

    def predict(self, X_holdout):
        return self.forest.predict(X_holdout)