# TODO check data classes
class ClassificationReport:
    def __init__(self, precision, recall, f1_score, accuracy, confusion_matrix):
        self.precision = precision
        self.recall = recall
        self.f1_score = f1_score
        self.accuracy = accuracy
        self.confusion_matrix = confusion_matrix

class ConfusionMatrix:
    def __init__(self, true_negative, false_positive, false_negative, true_positive):
        self.true_negative = true_negative
        self.false_positive = false_positive
        self.false_negative = false_negative
        self.true_positive = true_positive
    
