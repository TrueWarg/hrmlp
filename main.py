import os
from flask import Flask, request, redirect, url_for
from werkzeug.utils import secure_filename
import api.utils as utils 
import api.static.res as apires
import api.static.constants as apiconstants

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = apiconstants.__DEFAULT_UPLOAD_PATH

@app.route('/')
def test():
    return "test"
# TODO Probably, can't extrat features names from trained model (feature_importances_ get only numbers),
# so need add name to request body...
@app.route('/trainedmodel/upload', methods = ['POST'])
def upload_trained_model():
    file = request.files['file']
    if file and utils.is_allowed_file(file.filename, apiconstants.__ALLOWED_TRAINED_MODEL_EXTENSIONS):
        filename = secure_filename(file.filename)
        # add generating id and uni filename or let uploaded file 
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return apires.UPLOAD_TRAINED_MODEL_SUCCESS_MESSAGE
    return apires.UPLOAD_TRAINED_MODEL_ERROR_MESSAGE

if __name__ == '__main__':
    app.run()
