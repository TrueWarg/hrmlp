from joblib import dump, load
from storage.dbwokrhelpers import TrainedModelDbWorkHelper
# TODO need safeful method save model in db

class TraindedModelStorage:
    def save_trained_model_file(self, trained_model_file):
        db_helper = TrainedModelDbWorkHelper()
        generated_id = "id333"
        db_helper.insert_or_replace(generated_id, trained_model_file)
        return generated_id
        
    def delete_trained_model(self, model_id):
        db_helper = TrainedModelDbWorkHelper()
        db_helper.delete_by_id(model_id)


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
