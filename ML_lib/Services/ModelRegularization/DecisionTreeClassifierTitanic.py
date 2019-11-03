from sklearn.tree import DecisionTreeClassifier
from .Configuration import Configuration
from .ModelRegularizationService import ModelRegularizationService



class DecisionTreeClassifierTitanic:

    def __init__ (self):
        self.configuration = Configuration()
        self.name_estymator = "DecisionTreeClassifier"
        self.RegService = ModelRegularizationService(DecisionTreeClassifier(random_state=42), self.name_estymator)


    def approximation(self, train_strat_num_titanic):
        X, y = self.RegService.get_X_and_y_titanic(train_strat_num_titanic)

        param_grid_list = [
            {# default values
                'max_depth': [None], # int / None
                'splitter': ["best", 'random'], # best, random
                'min_samples_split': [2, 10, 100], # int / float
                'min_samples_leaf': [1, 10, 100], # int / float
                'min_weight_fraction_leaf': [0., 0.1, 0.5, 1.0],
                'max_features': [None, 'auto'], # int/float/auto/sqrt/log2/None
                'max_leaf_nodes': [None], # int / None
                'min_impurity_decrease': [0.],
                 'presort': [False]
            },
        ]

        print(f"{self.name_estymator} grid")
        dataframes = []
        for param_grid in param_grid_list:
            cv_results = self.RegService.get_grid_results(param_grid, X, y)
            result_grid_df = self.RegService.get_and_save_result_grid_dataframe(param_grid, cv_results, self.name_estymator)
            dataframes.append(result_grid_df)
        return dataframes


    def approximation_random_grid(self, train_strat_num_titanic, n=100):
        X, y = self.RegService.get_X_and_y_titanic(train_strat_num_titanic)
        param_grid_list = [
            {# default values
                'max_depth': [None, 2, 10, 20, 50], # int / None
                'splitter': ["best", 'random'], # best, random
                'min_samples_split': [2, 10, 100], # int / float
                'min_samples_leaf': [1, 10, 100], # int / float
                'min_weight_fraction_leaf': [0., 0.1, 0.5, 0.3],
                'max_features': [None, 'auto', 'sqrt', 'log2', 1, 5, 10, 0.1, 0.5, 0.8], # int/float/auto/sqrt/log2/None
                'max_leaf_nodes': [None, 2, 20, 40], # int / None
                'min_impurity_decrease': [0., 1e-5, 1e-6, 1e-7],
                 'presort': [False, True]
            },
        ]

        print(f"{self.name_estymator} random grid")
        dataframes = []

        for param_grid in param_grid_list:
            cv_results = self.RegService.get_random_grid_results(param_grid, X, y, n)
            result_grid_df = self.RegService.get_and_save_result_grid_dataframe(param_grid, cv_results, f"{self.name_estymator}_random_grid")
            dataframes.append(result_grid_df)
        
        return dataframes
    