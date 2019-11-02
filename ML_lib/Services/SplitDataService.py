import pandas as pd
from IPython.core.display import display
from sklearn.model_selection import StratifiedShuffleSplit



class SplitDataService:

    def __init__ (self):
        pass

    def split_train_test_strat(self, dataframe, column_strat):
        split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

        for train_index, test_index in split.split(dataframe, dataframe[column_strat]):
            strat_train_set = dataframe.loc[train_index]
            strat_test_set = dataframe.loc[test_index]
        return strat_train_set, strat_test_set

    

