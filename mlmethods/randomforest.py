
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

def get_trained_random_forest_classifier_search_cv(
    X_train,
    y_train,
    forest_grid_params,
    stratified_k_fold_n_splits=DEFAULT_STRATIFIED_N_SPLITS,
    n_estimators=DEFAULT_RANDOM_FOREST_ESTIMATORS_COUNT,
    stratified_k_fold_shuffle=True,
    use_out_of_bug_score=True,
    n_jobs=DEFALULT_N_JOBS,
    verbose=True
    ):
    stratified_k_fold=StratifiedKFold(
        n_splits=stratified_k_fold_n_splits, 
        shuffle=stratified_k_fold_shuffle)

    random_forest_grid = GridSearchCV(
        estimator=RandomForestClassifier(n_estimators=n_estimators),
        # TODO check on opportunity set n_estimator in params of RandomForestClassifier    
        param_grid=forest_grid_params, 
        n_jobs=n_jobs, 
        cv=stratified_k_fold, 
        verbose=verbose)

    random_forest_grid.fit(X_train, y_train)
    return random_forest_grid

def get_random_forest_classifier_grid_params(
    trees_max_depth_range,
    top_max_features_range,
    min_samples_leaf_range
    ):
    return {
        RANDOM_FOREST_PARAM_MAX_DEPTH : trees_max_depth_range,
        RANDOM_FOREST_PARAM_MAX_FEATURES : top_max_features_range,
        RANDOM_FOREST_PARAM_MIN_SAMPLES_LEAF : min_samples_leaf_range
    }


RANDOM_FOREST_PARAM_MAX_DEPTH = 'max_depth'
RANDOM_FOREST_PARAM_MAX_FEATURES = 'max_features'
RANDOM_FOREST_PARAM_MIN_SAMPLES_LEAF = 'min_samples_leaf'
DEFAULT_RANDOM_FOREST_ESTIMATORS_COUNT = 10

DEFALULT_N_JOBS = 1
DEFAULT_STRATIFIED_N_SPLITS = 5