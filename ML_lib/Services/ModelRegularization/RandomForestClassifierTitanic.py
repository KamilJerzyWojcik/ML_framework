from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
import json
from IPython.core.display import display
import pandas as pd
import time
from .Configuration import Configuration


class RandomForestClassifierTitanic:

    def __init__ (self):
        self.configuration = Configuration()
        self.randomForestClassifier = RandomForestClassifier(random_state=42)     

    def approximation_1(self, train_strat_num_titanic):
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

        print("RandomForestClassifier")
        dataframes = []

        for param_grid in param_grid_list:
            param_show = param_grid.copy()
            param_show[self.configuration.scoring] = 'prediction'

            for key in list(param_show.keys()):
                param_show[key] = str(param_show[key])

            cv_results = self.get_grid_results(param_grid, X, y)
            df = self.save_and_show_results(cv_results, param_show)
            dataframes.append(df)
        
        return dataframes
            
    
    def save_and_show_results(self, cv_results, param_show):
        display_result = []
        display_result.append(param_show)
        for scoring, params in sorted(zip(cv_results["mean_test_score"], cv_results["params"]), key=lambda x: x[0], reverse=True):
            params[self.configuration.scoring] = scoring
            display_result.append(params)
        result_df = pd.read_json(json.dumps(display_result))
        display(result_df)
        t = time.time()
        result_df.to_csv(f"RandomTreeClasifier/RandomTreeClasifier_{t}.csv", index=False)
        return result_df


    def get_grid_results(self, param_grid, X, y):
        gridSearchCV = GridSearchCV(
                self.randomForestClassifier, 
                [param_grid],
                cv=3,
                scoring=self.configuration.scoring, 
                return_train_score=True
            )
        gridSearchCV.fit(X, y)
        return gridSearchCV.cv_results_
    

    def get_X_and_y(self, train_strat_num_titanic):
        X_train = train_strat_num_titanic.drop(columns=['Survived', 'PassengerId'], axis=1)
        y_train = train_strat_num_titanic['Survived']
        return X_train, y_train
