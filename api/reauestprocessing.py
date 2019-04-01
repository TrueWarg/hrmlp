from flask import request
from api.static.constants import *
from api.utils import *
from api.response import *
from mlmethods.training import *

def process_training_request(request, trainer):
    data_set = request.files[DATA_SET_REQUEST_PARAM]
    if data_set and is_allowed_file(data_set.filename, ALLOWED_DATA_SET_FILE_EXTENSIONS):
        target = request.form[TRAINING_TARGET_FEATURE_REQUEST_PARAM]
        report = process_default_classifier_training(data_set, target, trainer)
        response = classifiaction_metrics_response(report)
        return response