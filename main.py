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
from storage.database import db_session, init_db
from storage.trainedmodelstorage import save_trained_model

app = Flask(__name__)
app.config[const.UPLOAD_FOLDER_CONFIG_PARAM] = const.DEFAULT_UPLOAD_PATH

# ------------------------------------------
@app.route('/trainedmodel/upload', methods = ['POST'])
def upload_trained_model():
    trained_model_file = request.files[const.UPLOAD_TRAINED_MODEL_REQUEST_PARAM]
    if trained_model_file and is_allowed_file(trained_model_file.filename, const.ALLOWED_TRAINED_MODEL_FILE_EXTENSIONS):
        #filename = secure_filename(trained_model_file.filename)
        # trainedModelFile.save(os.path.join(app.config[const.UPLOAD_FOLDER_CONFIG_PARAM], filename))
        generated_id = save_trained_model(trained_model_file)
        return generated_id
    return UPLOAD_TRAINED_MODEL_ERROR_MESSAGE

# ------------------Trees--------------------------
@app.route('/mlmethods/training/decisiontree', methods = ['POST'])
def train_decision_tree():
    response = process_training_request(request, DefaultTrainersFacroty().create_decision_tree_trainer())
    return response

@app.route('/prediction/decisiontree', methods = ['POST'])
def decisiontree_predict():
    # TODO make common solution (f.e. array) 
    features = {
        'satisfaction_level' : [request.form['satisfaction_level']],      
        'last_evaluation' : [request.form['last_evaluation']],
        'number_project' : [request.form['number_project']],
        'average_montly_hours' : [request.form['average_montly_hours']],
        'time_spend_company' : [request.form['time_spend_company']],
        'Work_accident' : [request.form['Work_accident']],
        'promotion_last_5years' : [request.form['promotion_last_5years']]
    }
    class_, proba = clf.get_prediction(features)
    return jsonify(
        willLeave = str(class_[0] == 1),
        leavingProbability = str(proba[0][1])
    )
    return "Prediction error"

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
    return "test"

@app.teardown_appcontext
def shutdown_session(exception=None):
    db_session.remove()

if __name__ == '__main__':
    app.run()
    init_db()
