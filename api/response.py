from flask import jsonify
from api.static.constants import CONFUSION_MATRINX_TNR, CONFUSION_MATRINX_FPR, CONFUSION_MATRINX_FNR, CONFUSION_MATRINX_TPR 

def classifiaction_metrics_response(report):
    return jsonify(
        precision = report.precision,
        recall = report.recall,
        f1Score = report.f1_score,
        accuracy = report.accuracy,
        confusionMatrix = serialize_confusion_matrix(report.confusion_matrix)
    )

def complete_training_response():
    return jsonify(
        success = True
    )

def serialize_confusion_matrix(confusion_matrix):
    return {
        CONFUSION_MATRINX_TNR : int(confusion_matrix.true_negative),
        CONFUSION_MATRINX_FPR : int(confusion_matrix.false_positive),
        CONFUSION_MATRINX_FNR : int(confusion_matrix.false_negative),
        CONFUSION_MATRINX_TPR : int(confusion_matrix.true_positive)
    }
