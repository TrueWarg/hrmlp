import pandas as pd
import mlmethods.randomforest as randomforest 
import mlmethods.decisiontrees as tree 
import storage.modelstorage as storage
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, confusion_matrix
from sklearn.model_selection import train_test_split
from mlmethods.constatns import TEST_SAMPLES_SIZE
from mlmethods.entities.metrics import ClassificationReport, ConfusionMatrix

def default_random_forest(data_set, target):
    X, y = __get_X_to_y(data_set, target)
    X_train, X_holdout, y_train, y_holdout = train_test_split(X, y, test_size=TEST_SAMPLES_SIZE)
    forest = randomforest.train_simple_random_forest(X, y)
    y_predicted = forest.predict(X_holdout)
    # TODO Add generation uni model id 
    model_id = 'hrml_test'
    storage.save_to_storage_by_id(random_forest_grid, model_id)
    report = ClassificationReport(
        precision = precision_score(y_holdout, y_predicted),
        recall = recall_score(y_holdout, y_predicted),
        f1_score = f1_score(y_holdout, y_predicted),
        accuracy = accuracy_score(y_holdout, y_predicted),
        confusion_matrix = __convert_confusion_matrix_to_object(confusion_matrix(y_holdout, y_predicted).ravel())
    )
    return report

def get_default_tree_with_report(data_set, target):
    X, y = __get_X_to_y(data_set, target)
    X_train, X_holdout, y_train, y_holdout = train_test_split(X, y, test_size=TEST_SAMPLES_SIZE)
    tree_grid = tree.get_trained_decision_tree_classifier_search_cv(X_train, y_train)
    y_predicted = tree_grid.predict(X_holdout)
    # TODO Add generation uni model id 
    model_id = 'hrml_test'
    storage.save_to_storage_by_id(tree_grid, model_id)
    report = ClassificationReport(
        precision = precision_score(y_holdout, y_predicted),
        recall = recall_score(y_holdout, y_predicted),
        f1_score = f1_score(y_holdout, y_predicted),
        accuracy = accuracy_score(y_holdout, y_predicted),
        confusion_matrix = __convert_confusion_matrix_to_object(confusion_matrix(y_holdout, y_predicted).ravel())
    )
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

def __convert_confusion_matrix_to_object(confusion_matrix_list):
    return ConfusionMatrix(
        true_negative = confusion_matrix_list[0],
        false_positive = confusion_matrix_list[1],
        false_negative = confusion_matrix_list[2],
        true_positive = confusion_matrix_list[3]
    )

