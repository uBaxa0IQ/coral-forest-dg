"""Canonicalize input null scalars before fitting or transforming any statistics."""
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from nir_dg.preprocessing import make_preprocessor as historical_preprocessor


def canonical_frame(frame):
    if not isinstance(frame,pd.DataFrame):
        raise TypeError("A named DataFrame schema is required")
    result=frame.copy()
    for column in result:
        if pd.api.types.is_numeric_dtype(result[column]):
            result[column]=result[column].astype(np.float64)
        else:
            values=result[column].astype(object)
            result[column]=values.where(pd.notna(values),np.nan)
    return result


def make_preprocessor(frame):
    canonical=canonical_frame(frame)
    return Pipeline([("canonical_nulls",FunctionTransformer(canonical_frame,validate=False,feature_names_out="one-to-one")),("source_fitted_transform",historical_preprocessor(canonical))])
