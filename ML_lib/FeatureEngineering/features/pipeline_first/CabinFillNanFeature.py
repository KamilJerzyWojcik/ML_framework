from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


class CabinFillNanFeature(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, df, y=None):
        return self

    def transform(self, df_mod, y=None):
        df_mod = pd.DataFrame(df_mod)
        df_mod['cabin_norm'] = df_mod['Cabin'].map(lambda x: 'missing' if str(x) == 'nan' else x[0] )
        df_mod = df_mod.drop(['Cabin'], axis=1)
        return df_mod