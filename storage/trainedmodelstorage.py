from joblib import dump, load
from storage.models.trainedmodels import TrainedModelDb
from storage.database import db
# TODO need safeful method save model in db

class TraindedModelStorage:
    
    def save_trained_model(self, model_name, filepath, user_id):
        generated_id = "id333"
        model_db = TrainedModelDb(
            id = generated_id,
            name = model_name,
            filepath = filepath,
            user_id = user_id
        )
        session = db.session()
        session.add(model_db)
        session.commit()
        return generated_id
        
    def delete_trained_model(self, model_id):
        return TrainedModelDb.query.filter(TrainedModelDb.id == model_id).delete()

    def get_all_by_user_id(self, user_id):
        return TrainedModelDb.query.filter(TrainedModelDb.id == user_id).all()



def save_to_storage_by_id(model_id, trained_model):
    filename = __get_file_name__(model_id)
    dump_result = dump(trained_model, filename)
    return dump_result

def get_from_storage_by_id(model_id):
    filename =  __get_file_name__(model_id)
    load_result = load(filename) 
    return load_result

def get_from_storage_by_file_name(filename):
    load_result = load(filename)
    return load_result

def __get_file_name__(variable_part):
    return __DEFAULT_PATH  + str(variable_part) + __FILE_EXTENSION
    
__TRAINED_MODEL_FILE_NAME_PREFIX = 'trained_model'
__FILE_EXTENSION = '.joblib'
# testing dirictory 
__DEFAULT_PATH = 'D:\\trainedmodels'
