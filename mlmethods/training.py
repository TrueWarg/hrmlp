import pandas as pd
from storage.trainedmodelstorage import TraindedModelStorage, convert_id_to_file_path
from storage.featurestorage import FeatureNamesStorage
from storage.models.trainedmodels import TrainedModelDb
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, confusion_matrix
from sklearn.model_selection import train_test_split
from mlmethods.constatns import TEST_SAMPLES_SIZE
from mlmethods.entities.metrics import ClassificationReport, ConfusionMatrix
import uuid

def process_default_classifier_training(data_set, target, trainer):
    X, y = __get_X_to_y(data_set, target)
    X_train, X_holdout, y_train, y_holdout = train_test_split(X, y, test_size=TEST_SAMPLES_SIZE)
    classifier = trainer.get_trained_classifier(X_train, y_train)
    y_predicted = classifier.predict(X_holdout)
    generated_id = str(uuid.uuid4())
    trained_model_storage = TraindedModelStorage()
    feature_names_storage = FeatureNamesStorage()
    filepath = convert_id_to_file_path(generated_id)
    # TODO make method for name gen
    model_db = TrainedModelDb(generated_id, str(classifier), filepath, user_id='lek')
    trained_model_storage.save_trained_model_as_file(classifier.child_classifier, filepath)
    trained_model_storage.save_trained_model(model_db)
    feature_names_storage.save_feature_names(X.columns, generated_id)
    # some classifier return real numbers
    y_predicted_int = list(map(lambda number: int(round(number)), y_predicted))
    report = ClassificationReport(
        precision = precision_score(y_holdout, y_predicted_int),
        recall = recall_score(y_holdout, y_predicted_int),
        f1_score = f1_score(y_holdout, y_predicted_int),
        accuracy = accuracy_score(y_holdout, y_predicted_int),
        confusion_matrix = __convert_confusion_matrix_to_object(confusion_matrix(y_holdout, y_predicted_int).ravel())
    )
    return report

def __binarize_object_features(data_frame, target):
    df_dummies = pd.get_dummies(data_frame, columns=data_frame.columns[data_frame.dtypes == 'object']).drop(target, axis=1)
    df_dummies.columns = [column_name.lower().replace(' ', '') for column_name in df_dummies.columns] 
    return df_dummies

def __drop_object_features(data_frame, target):
    data_frame.drop(target, axis=1, inplace=True)
    data_frame.drop(data_frame.columns[data_frame.dtypes == 'object'], axis=1, inplace=True)
    data_frame.columns = [column_name.lower().replace(' ', '') for column_name in data_frame.columns] 
    return data_frame

def __get_X_to_y(data_set, target):
    df = pd.read_csv(data_set)
    y = df[target]
    X = __drop_object_features(df, target)
    return X, y

def __convert_confusion_matrix_to_object(confusion_matrix_list):
    return ConfusionMatrix(
        true_negative = confusion_matrix_list[0],
        false_positive = confusion_matrix_list[1],
        false_negative = confusion_matrix_list[2],
        true_positive = confusion_matrix_list[3]
    )

