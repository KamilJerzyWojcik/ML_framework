from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


class EmbarkedFillNanFeature(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, df, y=None):
        return self

    def transform(self, df_mod, y=None):
        df_mod = pd.DataFrame(df_mod)
        df_mod['Embarked'] = df_mod['Embarked'].map(lambda x: 'missing' if str(x) == 'nan' else x[0])
        return df_mod