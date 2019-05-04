from storage.trainedmodelstorage import TraindedModelStorage
from storage.featurestorage import FeatureNamesStorage, extract_feature_names
from prediction.entities.results import PredictionResult
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
import xgboost as xgb
import pandas as pd

class Predictor:
    def get_prediction(self, values_file, model_id):
        trained_model_storage = TraindedModelStorage()
        X = self.__get_ordered_samples(values_file, model_id)
        trained_model_object = trained_model_storage.get_trained_model_object(model_id)
        self.__predict(trained_model_object, X)

# It's very specific method for getting prediction of object from extra libraries 
# (f.e. xgboost, skikit-learn). Can remove it, but then will be necessary 
# to dumb BaseClassifier after default training and to force client send type of prediction in upload method...
# Or need to come up with something else
    def __predict(self, trained_model_object, X): 
        if isinstance(trained_model_object, (GridSearchCV, LogisticRegressionCV, RandomForestClassifier)):
            values_to_proba = list(zip(trained_model_object.predict(X), trained_model_object.predict_proba(X)))
        elif isinstance(trained_model_object, xgb.Booster):
            predicted = trained_model_object.predict(xgb.DMatrix(X))
            values_to_proba = list(zip(list(map(lambda item: int(round(item)), predicted)), predicted))
        else:
            raise Exception("Unknown extra predictor's class")
        print("%%%%" + str(values_to_proba))
        return values_to_proba

    def __get_ordered_samples(self, values_file, model_id):
        df = pd.read_csv(values_file)
        feature_names_storge = FeatureNamesStorage()
        feature_names = list(map(lambda db_item: db_item.name, feature_names_storge.get_feture_names_by_model_id(model_id)))
        df.drop(df.columns[df.dtypes == 'object'], axis=1, inplace=True)
        df.columns = [column_name.lower().replace(' ', '') for column_name in df.columns] 
        # TODO no need to organize each time, it is better to check on difference 
        # between received columns names and sotred feature names 
        # in addition, some category feature will be reduced, bacause while trining
        # there are all values, but not for prediction
        ordered = df[feature_names]
        return ordered