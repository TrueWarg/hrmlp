import os
import mlmethods.training as training
import prediction.classification as clf
import validation.featurevalidation as vld
from api.utils import *
from api.response import *
from api.static.constants import *
from api.static.res import *
from api.reauestprocessing import *
from flask import Flask, request, redirect, url_for, jsonify
from mlmethods.trainersfactories import DefaultTrainersFacroty
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config[UPLOAD_FOLDER_CONFIG_PARAM] = DEFAULT_UPLOAD_PATH

# ------------------------------------------
# TODO Probably, can't extrat features names from trained model (feature_importances_ get only numbers),
# so need add name to request body...
@app.route('/trainedmodel/upload', methods = ['POST'])
def upload_trained_model():
    trainedModelFile = request.files[UPLOAD_TRAINED_MODEL_REQUEST_PARAM]
    if trainedModelFile and is_allowed_file(trainedModelFile.filename, ALLOWED_TRAINED_MODEL_FILE_EXTENSIONS):
        filename = secure_filename(trainedModelFile.filename)
        # add generating id and uni filename or let uploaded file 
        trainedModelFile.save(os.path.join(app.config[UPLOAD_FOLDER_CONFIG_PARAM], filename))
        return UPLOAD_TRAINED_MODEL_SUCCESS_MESSAGE
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
    if vld.validate_featues(features):
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

#---------------------------------------------
@app.route('/')
def test():
    return "test"

if __name__ == '__main__':
    app.run()


