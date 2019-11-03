from sklearn.pipeline import Pipeline
from .features.pipeline_first.SexCategoryFeature import SexCategoryFeature
from .features.pipeline_first.EmbarkedCategoryFeature import EmbarkedCategoryFeature
from .features.pipeline_first.TitleAdderFeature import TitleAdderFeature
from .features.pipeline_first.TitleNormalizationFeature import TitleNormalizationFeature
from .features.pipeline_first.TitleNormalizationCategoryFeature import TitleNormalizationCategoryFeature
from .features.pipeline_first.CabinFillNanFeature import CabinFillNanFeature
from .features.pipeline_first.CabinNormalizationCategoryFeature import CabinNormalizationCategoryFeature
from .features.pipeline_first.FamilySizeAdderFeature import FamilySizeAdderFeature
from .features.pipeline_first.TicketCategoryFeature import TicketCategoryFeature
from .features.pipeline_first.DropIdAndLabel import DropIdAndLabel
from .features.pipeline_first.SingleAdderFeature import SingleAdderFeature
from .features.pipeline_first.FareLogFeature import FareLogFeature
from .features.pipeline_first.AgeFillNan import AgeFillNan
from .features.pipeline_first.AgeBinFeature import AgeBinFeature
from .features.pipeline_first.EmbarkedFillNanFeature import EmbarkedFillNanFeature

from sklearn.preprocessing import StandardScaler
from IPython.core.display import display
import pandas as pd

class PipelineService:

    def __init__ (self):
        pass

    def pipeline_first(self, df):

        pipeline = Pipeline([
            ('Sex as number Category', SexCategoryFeature()),
            ('Embarked fill Nan', EmbarkedFillNanFeature()),
            ('Embarked number Category', EmbarkedCategoryFeature()),
            ('Title add', TitleAdderFeature()),
            ('Title select popular', TitleNormalizationFeature()),
            ('Title normalization', TitleNormalizationCategoryFeature()),
            ('Cabin fill nan by missing', CabinFillNanFeature()),
            ('Cabin normalization', CabinNormalizationCategoryFeature()),
            ('Familysize add', FamilySizeAdderFeature()),
            ('Family add', TicketCategoryFeature()),
            ('Single add', SingleAdderFeature()),
            ('FareLog add', FareLogFeature()),
            ('Age Fill Nan by median', AgeFillNan()),
            ('Age segregation', AgeBinFeature())
        ])

        dataframe_transformed = pipeline.fit_transform(df)

        return dataframe_transformed

    def pipeline_first_num(self, dataframe):
        dataframe_transformed = self.pipeline_first(dataframe)
        dataframe_transformed_num = dataframe_transformed.select_dtypes(include=['float64', 'int'])
        
        cat_cols = ['Survived', 'PassengerId', 'sex_cat']

        pipeline = Pipeline([
            ('standardScaler', StandardScaler())
        ])
        data_to_scaled = dataframe_transformed_num.drop(columns=cat_cols, axis=1)
        data_to_scaled_cols = data_to_scaled.columns

        scaled_num_array = pipeline.fit_transform(data_to_scaled)
        scaled_num_df = pd.DataFrame(data=scaled_num_array, columns=list(data_to_scaled_cols))

        scaled_num_df[cat_cols] = dataframe_transformed_num[cat_cols]

        return scaled_num_df


