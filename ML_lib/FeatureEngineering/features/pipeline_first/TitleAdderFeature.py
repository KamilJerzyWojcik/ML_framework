from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


class TitleAdderFeature(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, df, y=None):
        return self

    def transform(self, df, y=None):
        df_mod = pd.DataFrame(df)
        df_mod['title'] = df_mod['Name'].map(lambda x: x.split(',')[1].split('.')[0].lower().strip())
        return df_mod