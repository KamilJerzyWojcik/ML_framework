from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
import json
from IPython.core.display import display
import pandas as pd
import time
from .Configuration import Configuration


class RandomForestClassifierTitanic:

    def __init__ (self):
        self.configuration = Configuration()
        self.estymator = RandomForestClassifier(random_state=42)
        self.name_estymator = "RandomForestClassifier"


    def approximation(self, train_strat_num_titanic):
        X, y = self.get_X_and_y(train_strat_num_titanic)

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
            param_show = self.get_param_to_show(param_grid.copy())
            cv_results = self.get_grid_results(param_grid, X, y)
            result_grid_df = self.show_results(cv_results, param_show)
            self.save_results(result_grid_df, self.name_estymator)
            dataframes.append(result_grid_df)
        
        return dataframes


    def get_grid_results(self, param_grid, X, y):
            gridSearchCV = GridSearchCV(
                    self.estymator, 
                    [param_grid],
                    cv=3,
                    scoring=self.configuration.scoring, 
                    return_train_score=True
                )
            gridSearchCV.fit(X, y)
            return gridSearchCV.cv_results_
    

    def approximation_random_grid(self, train_strat_num_titanic, n=100):
        X, y = self.get_X_and_y(train_strat_num_titanic)
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
            param_show = self.get_param_to_show(param_grid.copy())
            cv_results = self.get_random_grid_results(param_grid, X, y, n)
            result_grid_df = self.show_results(cv_results, param_show)
            self.save_results(result_grid_df, f"{self.name_estymator}_random_grid")
            dataframes.append(result_grid_df)
        
        return dataframes
    

    def get_random_grid_results(self, param_grid, X, y, n):
        randomGridSearchCV = RandomizedSearchCV(
            estimator=self.estymator, 
            param_distributions=param_grid,
            n_iter=n,
            cv=3,
            scoring=self.configuration.scoring, 
            return_train_score=True
        )
        randomGridSearchCV.fit(X, y)
        return randomGridSearchCV.cv_results_


    def get_param_to_show(self, param_show):
        param_show[self.configuration.scoring] = 'prediction'
        for key in list(param_show.keys()):
            param_show[key] = str(param_show[key])
        return param_show


    def show_results(self, cv_results, param_show):
        display_result = []
        display_result.append(param_show)
        for scoring, params in sorted(zip(cv_results["mean_test_score"], cv_results["params"]), key=lambda x: x[0], reverse=True):
            params[self.configuration.scoring] = scoring
            display_result.append(params)
        result_df = pd.read_json(json.dumps(display_result))
        display(result_df)
        return result_df


    def save_results(self, result_df, name):
        current_time = time.time()
        result_df.to_csv(f"RandomTreeClasifier/{name}_{current_time}.csv", index=False)


    def get_X_and_y(self, train_strat_num_titanic):
        X_train = train_strat_num_titanic.drop(columns=['Survived', 'PassengerId'], axis=1)
        y_train = train_strat_num_titanic['Survived']
        return X_train, y_train
