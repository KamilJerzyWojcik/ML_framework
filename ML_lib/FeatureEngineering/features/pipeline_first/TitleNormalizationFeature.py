from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


class TitleNormalizationFeature(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.popular_titles = ['mr', 'miss', 'mrs', 'master', 'rev', 'dr']

    def fit(self, df, y=None):
        return self

    def transform(self, df, y=None):
        df_mod = pd.DataFrame(df)
        df_mod['title_norm'] = df_mod['title'].map(lambda x: x if x in self.popular_titles else 'other')
        return df_mod