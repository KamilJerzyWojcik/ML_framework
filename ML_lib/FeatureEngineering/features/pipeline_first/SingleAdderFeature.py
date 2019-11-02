from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


class SingleAdderFeature(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, df, y=None):
        return self

    def transform(self, df, y=None):
        df_mod = pd.DataFrame(df)
        df_mod['single'] = df_mod['family_size'].apply(lambda x: 1 if x == 0 else 0)
        return df_mod