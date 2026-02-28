import pandas as pd
from pandas.api.types import (
    is_object_dtype,
    is_string_dtype,
    is_float_dtype,
    is_integer_dtype,
)

def detect_problem_type(y):
    """
    Robust detection of classification vs regression
    """

    # ✅ String or object target → classification
    if is_object_dtype(y) or is_string_dtype(y):
        return "classification"

    # ✅ Float target → regression
    if is_float_dtype(y):
        return "regression"

    # ✅ Integer with few unique values → classification
    if is_integer_dtype(y) and y.nunique() <= 20:
        return "classification"

    # ✅ Default → regression
    return "regression"
