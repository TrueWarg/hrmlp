from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import constatns as const

# this method for default training functionality 
def get_kneighbors_classifier_search_cv(x_train, y_train):
    knn_params = {const.KNN_PARAM_NEIGHBORS_COUNT: range(2, 3)}
    knn_pipe = Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsClassifier())])
    knn_grid=GridSearchCV(estimator=knn_pipe, 
    param_grid=knn_params, 
    cv=const.CROSS_VALIDATION_COUNT,
    scoring=const.GRID_SEARCH_CV_SCORING,
    n_jobs=const.JOBS_NUMBER)

    knn_grid.fit(x_train, y_train)
    return knn_grid
