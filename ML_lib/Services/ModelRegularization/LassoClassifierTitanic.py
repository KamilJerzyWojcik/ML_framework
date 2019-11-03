from sklearn.linear_model import Lasso
from .Configuration import Configuration
from .ModelRegularizationService import ModelRegularizationService


class LassoClassifierTitanic:

    def __init__ (self):
        self.configuration = Configuration()
        self.name_estymator = "LassoClassifier"
        self.RegService = ModelRegularizationService(Lasso(random_state=42), self.name_estymator)


    def approximation(self, train_strat_num_titanic):
        X, y = self.RegService.get_X_and_y_titanic(train_strat_num_titanic)

        param_grid_list = [
            {# default values
                'alpha': [1.0], # int od 0
                'fit_intercept': [True],
                'normalize': [False],
                'precompute': [False],
                'copy_X': [True],
                # 'max_iter': [10],
                # ' tol': [1e-4],
                'positive': [False],
                'selection': ['cyclic'] # 'cyclic', 'random'
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
                'alpha': [1.0], # float
                'fit_intercept': [True],
                'normalize': [False],
                'copy_X': [True],
                'max_iter': [], # int or empty
                'solver': ['cholesky'], #  {‘auto’, ‘svd’, ‘cholesky’, ‘lsqr’, ‘sparse_cg’, ‘sag’, ‘saga’}
            },
        ]

        print(f"{self.name_estymator} random grid")
        dataframes = []

        for param_grid in param_grid_list:
            cv_results = self.RegService.get_random_grid_results(param_grid, X, y, n)
            result_grid_df = self.RegService.get_and_save_result_grid_dataframe(param_grid, cv_results, f"{self.name_estymator}_random_grid")
            dataframes.append(result_grid_df)
        
        return dataframes
    

    
    