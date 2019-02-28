from flask import jsonify

def classifiaction_metrics_response(report):
    return jsonify(
        precision = report.precision,
        recall = report.recall,
        f1Score = report.f1_score,
        accuracy = report.accuracy,
        confusionMatrix = report.confusion_matrix
    )

def complete_training_response():
    return jsonify(
        success = True
    )