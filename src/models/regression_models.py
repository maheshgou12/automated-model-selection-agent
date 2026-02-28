from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


def get_regression_models():
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor()
    }
    return models



