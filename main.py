import os
from flask import Flask, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
import api.utils as utils 
import api.static.res as apires
import api.static.constants as apiconstants
import mlmethods.training as training
import prediction.prediction as prd
import validation.validation as vld

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = apiconstants.DEFAULT_UPLOAD_PATH

@app.route('/')
def test():
    return "test"

# TODO Probably, can't extrat features names from trained model (feature_importances_ get only numbers),
# so need add name to request body...
@app.route('/trainedmodel/upload', methods = ['POST'])
def upload_trained_model():
    trainedModelFile = request.files['trainedModel']
    if trainedModelFile and utils.is_allowed_file(trainedModelFile.filename, apiconstants.ALLOWED_TRAINED_MODEL_FILE_EXTENSIONS):
        filename = secure_filename(trainedModelFile.filename)
        # add generating id and uni filename or let uploaded file 
        trainedModelFile.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return apires.UPLOAD_TRAINED_MODEL_SUCCESS_MESSAGE
    return apires.UPLOAD_TRAINED_MODEL_ERROR_MESSAGE

@app.route('/mlmethods/training/decisiontree', methods = ['POST'])
def train_decision_tree():
    data_set = request.files['dataSet']
    if data_set and utils.is_allowed_file(data_set.filename, apiconstants.ALLOWED_DATA_SET_FILE_EXTENSIONS):
         # TODO Add validation object features if it needed for client 
        target = request.form['target']
        training.default_tree(data_set, target)
        return 'Complete'
    return 'Error'

@app.route('/prediction/decisiontree', methods = ['POST'])
def decisiontree_predict():
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
        class_, proba = prd.get_prediction(features)
        return jsonify(
            class_ = str(class_),
            proba = str(proba[0])
        )
    return "Prediction error"

if __name__ == '__main__':
    app.run()
