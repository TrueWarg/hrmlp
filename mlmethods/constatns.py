# tree's hyper params
MAX_DEPTH = 'max_depth'
BOTTOM_MAX_DEPTH_VALUE = 2
TOP_MAX_DEPTH_VALUE = 12

MAX_FEATURES = 'max_features'
BOTTOM_MAX_FEATURES_VALUE = 3
TOP_MAX_FEATURES_VALUE = 7

MIN_SAMPLES_LEAF = 'min_samples_leaf'
BOTTOM_MIN_SAMPLES_LEAF_VALUE = 1 
TOP_MIN_SAMPLES_LEAF_VALUE = 20

MIN_SAMPLES_SPLIT = 'min_samples_split'
BOTTOM_MIN_SAMPLES_SPLIT_VALUE = 2 
TOP_MIN_SAMPLES_SPLIT_VALUE = 40

#-----------------------------------

# cross validation params 
CROSS_VALIDATION_COUNT = 5
JOBS_NUMBER = -1
RANDOM_STATE = 42
GRID_SEARCH_CV_SCORING = 'f1'
STRATIFIED_K_FOLD_COUNT = 5
TEST_SAMPLES_SIZE = 0.3

#-----------------------------------

# knn's hyper params
KNN_PARAM_NEIGHBORS_COUNT = 'knn__n_neighbors'
TOP_NEIGHBORS_COUNT_VALUE = 15

#-----------------------------------

# random forest params
RANDOM_FORES_TOP_MAX_DEPTH_VALUE = 7
RANDOM_FOREST_ESTIMATORS_COUNT = 100
RANDOM_FOREST_OOB_SCORE = True
