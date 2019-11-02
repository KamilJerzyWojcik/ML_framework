import pandas as pd
from IPython.core.display import display
from sklearn.model_selection import train_test_split


class DataframeService:
    """
        - konstruktor przyjmuje zarówno ściezke jak i dataframe
        - get_dataframe: obiekt dataframe
        - get_shape: rozmiar danych
        - get_values: array z wartościami
    """
    def __init__ (self, data):
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)
        if isinstance(data, str):
            self.___df = pd.read_csv(data)
        elif isinstance(data, pd.DataFrame):
            self.___df = data
        else:
            raise Exception("incorrect data type")
        print(f"imported data: ", type(self.___df))
    
    def get_dataframe(self):
        return self.___df
    
    def get_train_test(self):
        return train_test_split(self.___df, test_size=0.2)
    
    def get_shape(self):
        return self.___df.shape
    
    def get_values(self):
        return self.___df.values

    def save_colums_to_csv(self, df, path):
        df.to_csv(f'output/{path}', index=False)

    

