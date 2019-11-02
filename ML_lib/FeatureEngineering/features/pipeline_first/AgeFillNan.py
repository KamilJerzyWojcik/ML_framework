from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np


class AgeFillNan(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, df, y=None):
        return self

    def transform(self, df_mod, y=None):
        df_mod = pd.DataFrame(df_mod)
        missing_ages = df_mod.groupby('title_norm')['Age'].agg([np.mean, np.median]).to_dict()['median']
        df_mod['Age'] = df_mod.apply( lambda x: x['Age'] if str(x['Age']) != 'nan' else missing_ages[x['title_norm']], axis=1)
        return df_mod
