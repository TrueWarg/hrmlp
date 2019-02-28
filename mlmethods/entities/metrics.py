# TODO check data classes
class ClassificationReport:
    def __init__(self, precision, recall, f1_score, accuracy, confusion_matrix):
        self.precision = precision
        self.recall = recall
        self.f1_score = f1_score
        self.accuracy = accuracy
        self.confusion_matrix = confusion_matrix
