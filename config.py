import os
import api.static.constants as const

basedir = os.path.abspath(os.path.dirname(__file__))

class Config(object):
    UPLOAD_FOLDER = os.path.join(basedir, 'trainedmodels')
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
        'sqlite:///' + os.path.join(basedir, 'reservehrmlp.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    LOG_TO_STDOUT = os.environ.get('LOG_TO_STDOUT')

