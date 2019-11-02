from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


class CabinNormalizationCategoryFeature(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        pass

    def fit(self, df, y=None):
        return self

    def transform(self, df, y=None):
        df_mod = pd.DataFrame(df)
        df_mod['cabin_norm_cat'] = pd.factorize( df_mod['cabin_norm'] )[0]
        return df_mod