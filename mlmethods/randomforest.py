
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import constatns as const

def get_trained_random_forest_classifier_search_cv(X_train, y_train):
    stratified_k_fold = StratifiedKFold(n_splits=const.STRATIFIED_K_FOLD_COUNT, shuffle=True)
    forest_params = {
        const.RANDOM_FOREST_HYPER_PARAM_MAX_DEPTH : const.MAX_DEPTH_VALUES_LIST, 
        const.RANDOM_FOREST_HYPER_PARAM_MAX_FEATURES : range(1, const.TOP_MAX_FEATURE_VALUE),
        const.RANDOM_FOREST_HYPER_PARAM_MIN_SAMPLES_LEAF : const.MIN_SAMPLES_LEAF_VALUES_LIST
    }
    
    random_forest_clf = RandomForestClassifier(
        n_estimators=const.RANDOM_FOREST_ESTIMATORS_COUNT, 
        n_jobs=const.JOBS_NUMBER, 
        oob_score=const.RANDOM_FOREST_OOB_SCORE
    )
    
    random_forest_grid = GridSearchCV(
        estimator=random_forest_clf, 
        param_grid=forest_params, 
        n_jobs=const.JOBS_NUMBER, 
        cv=stratified_k_fold, 
        scoring=const.GRID_SEARCH_CV_SCORING)  
        
    random_forest_grid.fit(X_train, y_train)
    return random_forest_grid