from joblib import dump, load

def save_to_storage(trained_model, model_id):
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
    return __TRAINED_MODEL_FILE_NAME_PREFIX__ + str(variable_part) + __FILE_EXTENSION__

__TRAINED_MODEL_FILE_NAME_PREFIX__ = 'trained_model'
__FILE_EXTENSION__ = '.joblib'