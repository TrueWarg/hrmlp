from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from mlmethods.basetrainers import DefaulTrainer
import mlmethods.constatns as const

class DefaultKNeighborsTrainer(DefaulTrainer):
    def get_trained_classifier(self, X_train, y_train):
        return self.__get_kneighbors_classifier_search_cv(X_train, y_train)

    # this method for default training functionality 
    def __get_kneighbors_classifier_search_cv(self, X_train, y_train):
        knn_params = {const.KNN_PARAM_NEIGHBORS_COUNT: range(2, 3)}
        knn_pipe = Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsClassifier())])
        knn_grid=GridSearchCV(estimator=knn_pipe, 
        param_grid=knn_params, 
        cv=const.CROSS_VALIDATION_COUNT,
        scoring=const.GRID_SEARCH_CV_SCORING,
        n_jobs=const.JOBS_NUMBER)

        knn_grid.fit(X_train, y_train)
        return knn_grid
