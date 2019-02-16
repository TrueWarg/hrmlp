# TODO delete it after refactoring request
def validate_featues(features):
    for _, value in features.items():
        if value == None:
            return False
    return True