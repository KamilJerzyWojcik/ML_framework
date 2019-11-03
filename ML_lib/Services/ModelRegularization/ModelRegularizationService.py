from .Configuration import Configuration
import pandas as pd
import time
from IPython.core.display import display
import json
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV


class ModelRegularizationService:
    def __init__(self, estymator, path_save):
        self.configuration = Configuration()
        self.estymator = estymator
        self.path_save = path_save

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
        result_df.to_csv(f"{self.path_save}/{name}_{current_time}.csv", index=False)


    def get_X_and_y_titanic(self, train_strat_num_titanic):
        X_train = train_strat_num_titanic.drop(columns=['Survived', 'PassengerId'], axis=1)
        y_train = train_strat_num_titanic['Survived']
        return X_train, y_train


    def get_param_to_show(self, param_show):
        param_show[self.configuration.scoring] = 'prediction'
        for key in list(param_show.keys()):
            param_show[key] = str(param_show[key])
        return param_show


    def get_and_save_result_grid_dataframe(self, param_grid, cv_results, name_estymator):
        param_show = self.get_param_to_show(param_grid.copy())
        result_grid_df = self.show_results(cv_results, param_show)
        self.save_results(result_grid_df, name_estymator)
        return result_grid_df


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


  

