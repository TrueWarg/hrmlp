import os
import mlmethods.training as training
import prediction.classification as clf
import api.static.constants as const
from api.utils import *
from api.static.res import UPLOAD_TRAINED_MODEL_ERROR_MESSAGE
from api.reauestprocessing import process_training_request
from flask import Flask, request, redirect, url_for, jsonify
from mlmethods.trainersfactories import DefaultTrainersFacroty
from werkzeug.utils import secure_filename
from storage.trainedmodelstorage import TraindedModelStorage
from storage.database import init_db

app = Flask(__name__)
app.config[const.UPLOAD_FOLDER_CONFIG_PARAM] = const.DEFAULT_UPLOAD_PATH
app.config[const.SQLALCHEMY_DATABASE_URI_CONFIG_PARAM] = const.DEFAULT_DB_DIRICTORY

with app.app_context():
    init_db(app)

# ------------------------------------------
@app.route('/trainedmodel/upload', methods = ['POST'])
def upload_trained_model():
    trained_model_file = request.files[const.UPLOAD_TRAINED_MODEL_REQUEST_PARAM]
    if trained_model_file and is_allowed_file(trained_model_file.filename, const.ALLOWED_TRAINED_MODEL_FILE_EXTENSIONS):
        #filename = secure_filename(trained_model_file.filename)
        # trainedModelFile.save(os.path.join(app.config[const.UPLOAD_FOLDER_CONFIG_PARAM], filename))
        generated_id = ""
        return generated_id
    return UPLOAD_TRAINED_MODEL_ERROR_MESSAGE

# ------------------Trees--------------------------
@app.route('/mlmethods/training/decisiontree', methods = ['POST'])
def train_decision_tree():
    response = process_training_request(request, DefaultTrainersFacroty().create_decision_tree_trainer())
    return response

# ----------------Forest--------------------

@app.route('/mlmethods/training/randomforest', methods = ['POST'])
def train_random_forest():
    response = process_training_request(request, DefaultTrainersFacroty().create_random_forest_trainer())
    return response

# ----------------KNN-----------------------

@app.route('/mlmethods/training/kneighbors', methods = ['POST'])
def train_KNeighbors():
    response = process_training_request(request, DefaultTrainersFacroty().create_KNeighbors_trainer())
    return response

# ----------------Gradient Boosting----------
@app.route('/mlmethods/training/gradientboosting', methods = ['POST'])
def train_gradient_boosting():
    response = process_training_request(request, DefaultTrainersFacroty().create_gradient_boosting_trainer())
    return response

#-----------------Logistic Regression--------
@app.route('/mlmethods/training/logisticregression', methods = ['POST'])
def train_logistic_regression():
    response = process_training_request(request, DefaultTrainersFacroty().create_logistic_regression_trainer())
    return response

#---------------------------------------------
@app.route('/')
def test():
    return str(TraindedModelStorage().get_all_by_user_id("admin"))

if __name__ == '__main__':
    app.run()

