from sklearn.linear_model import LogisticRegression
from .Configuration import Configuration
from .ModelRegularizationService import ModelRegularizationService


class LogisticRegresionClassifierTitanic:

    def __init__ (self):
        self.configuration = Configuration()
        self.estymator = LogisticRegression(random_state=42)
        self.name_estymator = "LogisticRegresionClassifier"
        self.RegService = ModelRegularizationService(LogisticRegression(random_state=42))


    def approximation(self, train_strat_num_titanic):
        X, y = self.RegService.get_X_and_y(train_strat_num_titanic)

        param_grid_list = [
            {# default values
                'penalty': ['l1'], # ['l1', 'l2', 'elasticnet', 'none']
                'dual': [False], # [True, False]
                'tol': [1e-4], # float 
                'C': [1.0], #float
                'fit_intercept': [True],
                'intercept_scaling': [1.0], # float
                'class_weight': [None], # dct {class_label: weight}
                'solver': ['liblinear'], # newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’
                'max_iter': [100], # int
                'multi_class': ['ovr'], # ‘ovr’, ‘multinomial’, ‘auto’
                'verbose': [0], # int
                'warm_start': [False],
                'l1_ratio': [None] # float, None
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
        X, y = self.RegService.get_X_and_y(train_strat_num_titanic)
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
    

    
    