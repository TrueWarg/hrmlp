from sqlalchemy import Column, String, Integer
from storage.database import db

# Sqllite not suppurts array, can use postgresql
class FeatureNamelDb(db.Model):
    __tablename__="feature_name"
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(200))
    model_id = Column(String(100))
    order = Column(Integer())

    def __init__(self, name, model_id, order):
        self.name = name
        self.model_id = model_id
        self.order = order


