from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


class FamilySizeAdderFeature(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, df, y=None):
        return self

    def transform(self, df, y=None):
        df_mod = pd.DataFrame(df)
        df_mod['family_size'] = df_mod.apply(lambda x: x['Parch'] + x['SibSp'], axis=1)
        return df_mod