from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


class DropIdAndLabel(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.columsDropped = ["PassengerId", "Survived"]

    def fit(self, df, y=None):
        return self

    def transform(self, df, y=None):
        df_mod = pd.DataFrame(df)
        df_mod = df_mod.drop(columns=self.columsDropped, axis=1)
        return df_mod