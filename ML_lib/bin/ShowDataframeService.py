import pandas as pd
from IPython.core.display import display


class ShowDataframeService:
    """
        wykresy
        + analiza danych:
        - dwa wymiary col1 : col2, labelka bez znaczenia
        - dwa wymiary z rownymi kolorami labelki
        - histogramy
        - raport pandas
        - wykres rozproszenia

        + analiza modelu:
        - wykres precyzji w funkcji progu decyzyjnego
        - wykres precyzji w funkcji pełnosci
        - wykres ROC dla wielu
        - confusion matrix
        - 
    """
    def __init__ (self, dataframe):
        if isinstance(dataframe, pd.DataFrame):
            self.__df = dataframe
        else:
            display("select dataframe")

    def head(self, rows = 10):
        display(self.__df.head(rows))
    
    def sample(self, rows = 10):
        display(self.__df.sample(rows))

    def shape(self):
        display(self.__df.shape)

    def describe(self):
        display(self.__df.describe())

    def null_values(self):
        display(self.__df.isnull().sum())
    

