from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


class AgeBinFeature(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.age_bins = [0, 1, 3, 5, 9, 15, 20, 40, 60, 100]

    def fit(self, df, y=None):
        return self

    def transform(self, df, y=None):
        df_mod = pd.DataFrame(df)
        df_mod['age_bin'] = pd.factorize( pd.cut(df_mod['Age'], bins=self.age_bins).astype(object) )[0]
        return df_mod