from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from mlmethods.basetrainers import DefaulTrainer
import mlmethods.constatns as const

class DefaultDecisionTreeTrainer(DefaulTrainer):
    def get_trained_classifier(self, X_train, y_train):
        return self.__get_trained_decision_tree_classifier_search_cv(X_train, y_train)
    
    # this methods for default training functionality 
    def __get_trained_decision_tree_classifier_search_cv(self, X_train, y_train):
        tree_params = {
            const.MAX_DEPTH : range(
                const.BOTTOM_MAX_DEPTH_VALUE, 
                const.TOP_MAX_DEPTH_VALUE
            ),
            const.MAX_FEATURES : range(
                const.BOTTOM_MAX_FEATURES_VALUE, 
                len(X_train.columns)
                )
            }
        tree_grid=GridSearchCV(
            estimator=DecisionTreeClassifier(), 
            param_grid=tree_params,
            cv=const.CROSS_VALIDATION_COUNT, 
            n_jobs=const.JOBS_NUMBER,
            scoring=const.GRID_SEARCH_CV_SCORING
        )
        tree_grid.fit(X_train, y_train)
        return tree_grid