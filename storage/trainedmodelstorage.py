import os
from joblib import dump, load
from storage.models.trainedmodels import TrainedModelDb
from storage.database import db
from config import Config
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
def convert_id_to_file_path(id):
    print(Config.UPLOAD_FOLDER)
    return os.path.join(Config.UPLOAD_FOLDER, id + __FILE_EXTENSION)

__FILE_EXTENSION = '.joblib'
