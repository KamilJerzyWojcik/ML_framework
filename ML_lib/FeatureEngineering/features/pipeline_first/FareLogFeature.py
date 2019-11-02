from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np


class FareLogFeature(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, df, y=None):
        return self

    def transform(self, df, y=None):
        df_mod = pd.DataFrame(df)
        df_mod['fare_log'] = np.log2( df_mod['Fare'] + 1 )
        return df_mod