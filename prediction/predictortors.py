from storage.trainedmodelstorage import TraindedModelStorage
from storage.featurestorage import FeatureNamesStorage, extract_feature_names
from prediction.entities.results import PredictionResult
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
import xgboost as xgb
import pandas as pd

def get_prediction(filled_form, model_id = 'hrml_test'):
    X = pd.DataFrame.from_dict(filled_form)
    model = storage.get_from_storage_by_id(model_id)
    class_ = model.predict(X)
    proba = model.predict_proba(X)
    return class_, proba

class Predictor:
    def get_prediction(self, values_file, model_id):
        trained_model_storage = TraindedModelStorage()
        feature_name_storage = FeatureNamesStorage()
        trained_model_object = trained_model_storage.get_trained_model_object(model_id)

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
        return values_to_proba


    def __get_ordered_samples(self, values_file, model_id):
        df = pd.read_csv(values_file)
        df_dummies = pd.get_dummies(df, columns=df.columns[df.dtypes == 'object'])
