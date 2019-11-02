from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


class TicketCategoryFeature(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, df, y=None):
        return self

    def transform(self, df, y=None):
        df_mod = pd.DataFrame(df)
        df_mod['ticket_cat'] = pd.factorize(df_mod['Ticket'])[0]
        return df_mod