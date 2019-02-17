import pandas as pd
import mlmethods.randomforest as randomforest 
import mlmethods.decisiontrees as tree 
import modelstorage.modelstorage as storage

def default_random_forest(data_set, target):
    print(str(data_set))
    df = pd.read_csv(data_set)
    y = df[target]
    X = __binarize_object_features(df, target)
    random_forest_grid = randomforest.get_trained_random_forest_classifier_search_cv(X, y)
    storage.save_to_storage_by_id(random_forest_grid, 'hrml_test')
    return 

def default_tree(data_set, target):
    print(str(data_set))
    df = pd.read_csv(data_set)
    y = df[target]
    X = __drop_object_features(df, target)
    tree_grid = tree.get_trained_decision_tree_classifier_search_cv(X, y)
    storage.save_to_storage_by_id(tree_grid, 'hrml_test')
    return tree_grid

def __binarize_object_features(data_frame, target):
    data_frame_dummies = pd.get_dummies(data_frame, columns=data_frame.columns[data_frame.dtypes == 'object']).drop(target, axis=1)
    return data_frame_dummies

def __drop_object_features(data_frame, target):
    data_frame.drop(target, axis=1, inplace=True)
    return data_frame.drop(data_frame.columns[data_frame.dtypes == 'object'], axis=1)
