from sklearn.ensemble import ExtraTreesClassifier
from .Configuration import Configuration
from .ModelRegularizationService import ModelRegularizationService



class ExtraTreesClassifierTitanic:

    def __init__ (self):
        self.configuration = Configuration()
        self.name_estymator = "ExtraTreesClassifier"
        self.RegService = ModelRegularizationService(ExtraTreesClassifier(random_state=42), self.name_estymator)


    def approximation(self, train_strat_num_titanic):
        X, y = self.RegService.get_X_and_y_titanic(train_strat_num_titanic)

        param_grid_list = [
            {# default values
                'n_estimators': [50], # int
                'max_depth': [10], # int / None
                'min_samples_split': [10], # int / float
                'min_samples_leaf': [1], # int / float
                'min_weight_fraction_leaf': [0.], # float
                'max_features': ["sqrt"], # int/float/auto/sqrt/log2/None
                'max_leaf_nodes': [None], # int / None
                'min_impurity_decrease': [0.], # float
                'bootstrap': [True],
                'oob_score': [False], # if bootstrap true
                'warm_start': [False],
                'criterion': ['gini'] # entropy, gini
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
                'n_estimators': [25, 50, 75], # int
                'max_depth': [10, 15, 20], # int / None
                'min_samples_split': [10, 15, 20], # int / float
                'min_samples_leaf': [1, 2, 5, 10], # int / float
                'min_weight_fraction_leaf': [0., 0.1, 0.5], # float
                'max_features': ["auto", "sqrt", 'log2', None, 0.1, 0.5, 0.7], # int/float/auto/sqrt/log2/None
                'max_leaf_nodes': [None], # int / None
                'min_impurity_decrease': [0.], # float
                'bootstrap': [True],
                'oob_score': [False], # if bootstrap true
                'warm_start': [False]
            },
        ]

        print(f"{self.name_estymator} random grid")
        dataframes = []

        for param_grid in param_grid_list:
            cv_results = self.RegService.get_random_grid_results(param_grid, X, y, n)
            result_grid_df = self.RegService.get_and_save_result_grid_dataframe(param_grid, cv_results, f"{self.name_estymator}_random_grid")
            dataframes.append(result_grid_df)
        
        return dataframes
    