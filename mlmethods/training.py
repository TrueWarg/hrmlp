import pandas as pd
import mlmethods.randomforest as randomforest 
import mlmethods.decisiontrees as tree 
import storage.modelstorage as storage
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, confusion_matrix
from sklearn.model_selection import train_test_split
from mlmethods.constatns import TEST_SAMPLES_SIZE
from mlmethods.entities.metrics import ClassificationReport

def default_random_forest(data_set, target):
    df = pd.read_csv(data_set)
    y = df[target]
    X = __binarize_object_features(df, target)
    random_forest_grid = randomforest.get_trained_random_forest_classifier_search_cv(X, y)
    # TODO Add generation uni model id 
    storage.save_to_storage_by_id(random_forest_grid, 'hrml_test')
    return storage

def train_default_tree(data_set, target):
    X, y = __get_X_to_y(data_set, target)
    tree_grid = tree.get_trained_decision_tree_classifier_search_cv(X, y)
    model_id = 'hrml_test'
    storage.save_to_storage_by_id(tree_grid, model_id)
    return tree_grid

def get_default_tree_with_report(data_set, target):
    X, y = __get_X_to_y(data_set, target)
    X_train, X_holdout, y_train, y_holdout = train_test_split(X, y, test_size=TEST_SAMPLES_SIZE)
    tree_grid = tree.get_trained_decision_tree_classifier_search_cv(X_train, y_train)
    y_predicted = tree_grid.predict(X_holdout)
    model_id = 'hrml_test'
    # TODO need to decide retrain on all data or save model which trained only part...
    # Probabaly, can use best params from GreedSearchCV and make simple training
    storage.save_to_storage_by_id(tree_grid, model_id)
    report = ClassificationReport(
        precision = precision_score(y_train, y_predicted),
        recall = recall_score(y_train, y_predicted),
        f1_score = f1_score(y_train, y_predicted),
        accuracy = accuracy_score(y_train, y_predicted),
        confusion_matrix = confusion_matrix(y_train, y_predicted))
    return report

def __binarize_object_features(data_frame, target):
    data_frame_dummies = pd.get_dummies(data_frame, columns=data_frame.columns[data_frame.dtypes == 'object']).drop(target, axis=1)
    return data_frame_dummies

def __drop_object_features(data_frame, target):
    data_frame.drop(target, axis=1, inplace=True)
    return data_frame.drop(data_frame.columns[data_frame.dtypes == 'object'], axis=1)

def __get_X_to_y(data_set, target):
    df = pd.read_csv(data_set)
    y = df[target]
    X = __binarize_object_features(df, target)
    return X, y

