from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

# Propably, it is  not necessary wrap
def get_trained_decision_tree_classifier_search_cv(
    X_train,  
    y_train, 
    tree_grid_params, 
    cross_validation_coun=DEFALULT_CROSS_VALIDATION_COUNT, 
    n_jobs=DEFALULT_N_JOBS,
    verbose=True):

    tree_grid=GridSearchCV(
        estimator=DecisionTreeClassifier(), 
        param_grid=tree_grid_params, 
        cv=cross_validation_coun, 
        n_jobs=n_jobs,
        verbose=verbose
    )

    tree_grid.fit(X_train, y_train)
    return tree_grid

# If will be make learning's functional for client, should add another important customizable params
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
DEFALULT_N_JOBS = 1
