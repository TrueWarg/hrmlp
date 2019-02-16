import os
from flask import Flask, request, redirect, url_for
from werkzeug.utils import secure_filename
import api.static.res as apires
import api.static.constants as apiconstants

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = apiconstants.__DEFAULT_UPLOAD_PATH

@app.route('/')
def test():
    return "test"

@app.route('/trainedmodel/upload', methods = ['POST'])
def upload_trained_model():
    file = request.files['file']
    if file and __is_allowed_file(file.filename, apiconstants.__ALLOWED_TRAINED_MODEL_EXTENSIONS):
        filename = secure_filename(file.filename)
        # add generating id and uni filename or let uploaded file 
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return apires.UPLOAD_TRAINED_MODEL_SUCCESS_MESSAGE
    return apires.UPLOAD_TRAINED_MODEL_ERROR_MESSAGE

def __is_allowed_file(filename, allowed_extensions):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in allowed_extensions

if __name__ == '__main__':
    app.run()
