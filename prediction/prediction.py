import modelstorage.modelstorage as storage
import pandas as pd

def get_prediction(filled_form, model_id = 'hrml_test'):
    X = pd.DataFrame.from_dict(filled_form)
    model = storage.get_from_storage_by_id(model_id)
    class_ = model.predict(X)
    proba = model.predict_proba(X)
    return class_, proba