from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


class EmbarkedCategoryFeature(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, df, y=None):
        return self

    def transform(self, df, y=None):
        df_mod = pd.DataFrame(df)
        df_mod['embarked_cat'] = pd.factorize( df_mod['Embarked'] )[0]
        return df_mod