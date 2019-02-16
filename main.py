import os
from flask import Flask, request, redirect, url_for
from werkzeug.utils import secure_filename
import api.utils as utils 
import api.static.res as apires
import api.static.constants as apiconstants
import mlmethods.randomforest as randomforest

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = apiconstants.DEFAULT_UPLOAD_PATH

if __name__ == '__main__':
    app.run()

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

@app.route('/mlmethods/training/randomforest', methods = ['POST'])
def train_randomforest():
    data_set = request.files['dataSet']
    if data_set and utils.is_allowed_file(data_set.filename, apiconstants.ALLOWED_DATA_SET_FILE_EXTENSIONS):
        #  TODO Add validation and binarization object features if it needed for client

