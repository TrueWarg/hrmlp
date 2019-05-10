from flask_sqlalchemy import SQLAlchemy
# Add mechanism for migrations 
db = SQLAlchemy()

def init_db(app):
    db.init_app(app) 
    db.create_all()
