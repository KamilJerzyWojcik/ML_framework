from sklearn.tree import DecisionTreeClassifier
from .Configuration import Configuration
from .ModelRegularizationService import ModelRegularizationService
from sklearn.svm import SVC



class SVCClassifierTitanic:

    def __init__ (self):
        self.configuration = Configuration()
        self.name_estymator = "SVCClassifierTitanic"
        self.RegService = ModelRegularizationService(SVC(random_state=42), self.name_estymator)


    def approximation(self, train_strat_num_titanic):
        X, y = self.RegService.get_X_and_y_titanic(train_strat_num_titanic)

        param_grid_list = [
            {# default values
                'C': [1.0, 5.0, 0.1, 0.01, 0.001, 10],
                'kernel': ['rbf', 'linear', 'poly', 'sigmoid'], # linear, poly, rbf, sigmoid, precomputed
                'degree': [3, 2, 1],
                'gamma': ['auto', 'scale'], # scale, auto
                'coef0': [0.0, 1.0, 0.1],
                'shrinking': [True, False],
                'probability': [False, True],
                'tol': [1e-3],
                'max_iter': [-1],
                'decision_function_shape': ['ovr', 'ovo'], # ovr, ovo
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
                'C': [1.0, 5.0, 0.1, 0.01, 0.001, 10, 50, 100],
                'kernel': ['rbf', 'linear', 'poly', 'sigmoid'], # linear, poly, rbf, sigmoid, precomputed
                'degree': [3, 2, 1, 4],
                'gamma': ['auto', 'scale'], # scale, auto
                'coef0': [0.0, 1.0, 0.1, 10, 0.01, 0.001],
                'shrinking': [True, False],
                'probability': [False, True],
                'tol': [1e-3, 1e-2],
                'max_iter': [-1],
                'decision_function_shape': ['ovr', 'ovo'], # ovr, ovo
            },
        ]

        print(f"{self.name_estymator} random grid")
        dataframes = []

        for param_grid in param_grid_list:
            cv_results = self.RegService.get_random_grid_results(param_grid, X, y, n)
            result_grid_df = self.RegService.get_and_save_result_grid_dataframe(param_grid, cv_results, f"{self.name_estymator}_random_grid")
            dataframes.append(result_grid_df)
        
        return dataframes
    