from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


class TitleNormalizationCategoryFeature(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, df, y=None):
        return self

    def transform(self, df, y=None):
        df_mod = pd.DataFrame(df)
        df_mod['title_norm_cat'] = pd.factorize(df_mod['title_norm'])[0]
        return df_mod