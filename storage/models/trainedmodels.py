from sqlalchemy import Column, String, Binary
from storage.database import db

class TrainedModelDb(db.Model):
    __tablename__="trained_models"
    id = Column(String(100), primary_key=True)
    name = Column(String(100))
    filepath = Column(String(100), unique=True)
    user_id = Column(String(100), unique=True)

    def __init__(self, id, name, filepath, user_id):
        self.id = id
        self.name = name
        self.filepath = filepath
        self.user_id = user_id
