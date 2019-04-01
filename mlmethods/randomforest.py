from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from mlmethods.basetrainers import DefaulTrainer
import mlmethods.constatns as const

class DefaultRandomForestTrainer(DefaulTrainer):
    def get_trained_classificator(self, X_train, y_train):
        return self.__train_simple_random_forest(X_train, y_train)

    # this method for default training functionality 
    def __train_simple_random_forest(self, X_train, y_train):
        forest = RandomForestClassifier(
            n_estimators=const.RANDOM_FOREST_ESTIMATORS_COUNT, 
            max_depth=const.RANDOM_FORES_TOP_MAX_DEPTH_VALUE,
            n_jobs=const.JOBS_NUMBER,
            oob_score=const.RANDOM_FOREST_OOB_SCORE
        )
        forest.fit(X_train, y_train)
        return forest