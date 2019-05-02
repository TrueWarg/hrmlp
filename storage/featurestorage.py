from storage.models.features import FeatureNamelDb
from storage.database import db
import pandas as pd

class FeatureNamesStorage:
    
    def save_feature_names(self, feature_names, model_id):
        session = db.session()
        for index, name in enumerate(feature_names):
            name_db = FeatureNamelDb(name=name, model_id=model_id, order=index)
            session.add(name_db)
        session.commit()

    def get_feture_names_by_model_id(self, model_id):
        return sorted(FeatureNamelDb.query.filter(FeatureNamelDb.model_id == model_id).all(), \
            key=lambda name: name.order)

#utils
def extract_feature_names(values_file):
    df = pd.read_csv(values_file)
    df_dummies = pd.get_dummies(df, columns=df.columns[df.dtypes == 'object'])
    formatted = list(map(lambda feature_name: feature_name.lower().replace(' ', ''), df_dummies.columns))
    return formatted
