import pandas as pd 
import randomforest 

def default_random_forest(dataSet, target):
    df = pd.read_csv(dataSet)
    y = df.dtop(target)
    X = __binarize_object_features(df, target)
    random_forest_grid = randomforest.get_trained_random_forest_classifier_search_cv(X, y)

def __binarize_object_features(data_frame, target):
    data_frame_dummies = pd.get_dummies(data_frame, columns=data_frame.columns[data_frame.dtypes == 'object']).drop(target, axis=1)
    return data_frame_dummies

def __drop_object_features(data_frame, target):
    data_frame 
