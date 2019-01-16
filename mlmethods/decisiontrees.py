from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

def get_trained_decision_tree_classifier_search_cv(
    X_train,  
    y_train, 
    tree_grid_params, 
    cross_validation_coun=DEFALULT_CROSS_VALIDATION_COUNT, 
    jobs_count=DEFALULT_JOBS_COUNT):

    tree_grid=GridSearchCV(estimator=DecisionTreeClassifier(), 
    param_grid=tree_grid_params, 
    cv=cross_validation_coun, 
    n_jobs=n_jobs,
    verbose=True)

    tree_grid.fit(X_train, y_train)
    return tree_grid

# hyper params 
TREE_CLS_PARAM_MAX_DEPTH = 'max_depth'
TREE_CLS_MAX_FEATURES = 'max_features'

# default tree's hyper params top values 
DEFALULT_TOP_MAX_DEPTH_VALUE = 10
DEFALULT_TOP_MAX_FEATURES_VALUE = 7 

# grid search cv default values 
DEFALULT_CROSS_VALIDATION_COUNT = 5
DEFALULT_JOBS_COUNT = 1
