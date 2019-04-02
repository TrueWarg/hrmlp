from flask import jsonify

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
        'trueNegative' : int(confusion_matrix.true_negative),
        'falsePositive' : int(confusion_matrix.false_positive),
        'falseNegative' : int(confusion_matrix.false_negative),
        'truePositive' : int(confusion_matrix.true_positive)
    }
