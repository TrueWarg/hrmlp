from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

def get_trained_decision_tree_classifier_search_cv(
    X_train,  
    y_train, 
    tree_grid_params, 
    cross_validation_coun=DEFALULT_CROSS_VALIDATION_COUNT, 
    jobs_count=DEFALULT_JOBS_COUNT,
    verbose=True):

    tree_grid=GridSearchCV(
        estimator=DecisionTreeClassifier(), 
        param_grid=tree_grid_params, 
        cv=cross_validation_coun, 
        n_jobs=jobs_count,
        verbose=verbose
    )

    tree_grid.fit(X_train, y_train)
    return tree_grid

def get_tree_classifier_grid_params(
    max_depth_range=range(1,DEFALULT_TOP_MAX_FEATURES_VALUE),
    top_max_features_range=(1, DEFALULT_TOP_MAX_FEATURES_VALUE)):

    tree_params = {}
    tree_params[TREE_CLS_PARAM_MAX_DEPTH] = max_depth_range
    tree_params[TREE_CLS_MAX_FEATURES] = top_max_features_range

    return tree_params

# hyper params 
TREE_CLS_PARAM_MAX_DEPTH = 'max_depth'
TREE_CLS_MAX_FEATURES = 'max_features'

# default tree's hyper params top values 
DEFALULT_TOP_MAX_DEPTH_VALUE = 10
DEFALULT_TOP_MAX_FEATURES_VALUE = 7 

# grid search cv default values 
DEFALULT_CROSS_VALIDATION_COUNT = 5
DEFALULT_JOBS_COUNT = 1
