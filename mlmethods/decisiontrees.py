from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
import mlmethods.constatns as const

# this methods for default training functionality 
def get_trained_decision_tree_classifier_search_cv(x_train, y_train):
    tree_params = {
        const.MAX_DEPTH : range(
            const.BOTTOM_MAX_DEPTH_VALUE, 
            const.TOP_MAX_DEPTH_VALUE
        ),
        const.MAX_FEATURES : range(
            const.BOTTOM_MAX_FEATURES_VALUE, 
            len(x_train.columns)
        )
    }
    tree_grid=GridSearchCV(
        estimator=DecisionTreeClassifier(), 
        param_grid=tree_params,
        cv=const.CROSS_VALIDATION_COUNT, 
        n_jobs=const.JOBS_NUMBER,
        scoring=const.GRID_SEARCH_CV_SCORING)

    tree_grid.fit(x_train, y_train)
    return tree_grid