from sqlalchemy import Column, String, Binary
from storage.database import Base

class TrainedModel(Base):
    __tablename__="trained_models"
    id = Column(String(60), primary_key=True)
    binary_form = Column(Binary)

    def __init__(self, id, binary_form):
        self.id = id
        self.binary_form = binary_form
