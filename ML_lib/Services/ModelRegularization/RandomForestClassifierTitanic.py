from sklearn.ensemble import RandomForestClassifier
from .Configuration import Configuration
from .ModelRegularizationService import ModelRegularizationService



class RandomForestClassifierTitanic:

    def __init__ (self):
        self.configuration = Configuration()
        self.name_estymator = "RandomForestClassifier"
        self.RegService = ModelRegularizationService(RandomForestClassifier(random_state=42), self.name_estymator)


    def approximation(self, train_strat_num_titanic):
        X, y = self.RegService.get_X_and_y_titanic(train_strat_num_titanic)

        param_grid_list = [
            {# default values
                'n_estimators': [10], # int
                'max_depth': [10], # int / None
                'min_samples_split': [20], # int / float
                'min_samples_leaf': [1], # int / float
                'min_weight_fraction_leaf': [0.], # float
                'max_features': ["auto"], # int/float/auto/sqrt/log2/None
                'max_leaf_nodes': [None], # int / None
                'min_impurity_decrease': [0.], # float
                'bootstrap': [True],
                'oob_score': [False], # if bootstrap true
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
                'n_estimators': [1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15, 16], # int
                'max_depth': [1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15, 16], # int / None
                'min_samples_split': [10, 11, 12, 13, 14, 15, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30], # int / float
                'min_samples_leaf': [1, 2, 3, 4, 0.001, 0.01, 0.02, 0.03, 0.04, 0.1], # int / float
                'min_weight_fraction_leaf': [0., 0.01, 0.0001], # float
                'max_features': ["auto", "sqrt", "log2", None], # int/float/auto/sqrt/log2/None
                'max_leaf_nodes': [None, 2, 5, 10], # int / None
                'min_impurity_decrease': [0., 0.001, 0.01, 0.1], # float
                'bootstrap': [True, False],
                'oob_score': [False], # if bootstrap true
            },
        ]

        print(f"{self.name_estymator} random grid")
        dataframes = []

        for param_grid in param_grid_list:
            cv_results = self.RegService.get_random_grid_results(param_grid, X, y, n)
            result_grid_df = self.RegService.get_and_save_result_grid_dataframe(param_grid, cv_results, f"{self.name_estymator}_random_grid")
            dataframes.append(result_grid_df)
        
        return dataframes
    