from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def get_trained_kneighbors_classifier_search_cv(
    X_train, 
    y_train,
    knn_grid_params,
    cross_validation_coun=DEFALULT_CROSS_VALIDATION_COUNT, 
    n_jobs=DEFALULT_N_JOBS,
    verbose=True
    ):
    knn_pipe = Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsClassifier(n_jobs=-1))])
    
    knn_grid=GridSearchCV(estimator=knn_pipe, 
    param_grid=knn_grid_params, 
    cv=DEFALULT_CROSS_VALIDATION_COUNT, 
    n_jobs=n_jobs,
    verbose=verbose)

    knn_grid.fit(X_train, y_train)
    return knn_grid


# grid search cv default values. 
# TODO check on namespace coflict if import 
DEFALULT_CROSS_VALIDATION_COUNT = 5
DEFALULT_N_JOBS = 1