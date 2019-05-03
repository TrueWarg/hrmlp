from joblib import dump, load
from storage.models.trainedmodels import TrainedModelDb
from storage.database import db

class TraindedModelStorage:
    def save_trained_model(self, model_db):
        session = db.session()
        session.add(model_db)
        session.commit()

    def save_trained_model_as_file(self, trained_model, filepath):
        dump(trained_model, filepath)

    def get_trained_model_object(self, model_id):
        db_model = TrainedModelDb.query.filter(TrainedModelDb.id == model_id).first()
        return load(db_model.filepath)

    def delete_trained_model(self, model_id):
        return TrainedModelDb.query.filter(TrainedModelDb.id == model_id).delete()

    def get_all_by_user_id(self, user_id):
        return TrainedModelDb.query.filter(TrainedModelDb.user_id == user_id).all()

# Utils     
# Now only windows test and 
# TODO make segmentation '-' replace on '\\'
def convert_id_to_file_path(id):
    return __DEFAULT_PATH + '\\' + id + __FILE_EXTENSION

__TRAINED_MODEL_FILE_NAME_PREFIX = 'trained_model'
__FILE_EXTENSION = '.joblib'
# testing dirictory 
__DEFAULT_PATH = 'D:\\trainedmodels'
