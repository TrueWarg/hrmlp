from mlmethods.decisiontrees import DefaultDecisionTreeTrainer
from mlmethods.knearneighbour import DefaultKNeighborsTrainer
from mlmethods.randomforest import DefaultRandomForestTrainer
from mlmethods.gradientboosting import DefaultGradientBoostingTrainer

class DefaultTrainersFacroty():
    def create_decision_tree_trainer(self):
        return DefaultDecisionTreeTrainer()

    def create_KNeighbors_trainer(self):
        return DefaultKNeighborsTrainer()

    def create_random_forest_trainer(self):
        return DefaultRandomForestTrainer()

    def create_gradient_boosting_trainer(self):
        return DefaultGradientBoostingTrainer()

