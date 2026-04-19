"""rotten-tomatoes-forecasting: Poisson-binomial forecasting for Kalshi RT Tomatometer markets."""

from rotten_tomatoes_forecasting.edge import compute_edge, naive_lambda, naive_p_fresh
from rotten_tomatoes_forecasting.features import extract_lambda_features
from rotten_tomatoes_forecasting.lambda_model import (
    LambdaPrediction,
    LambdaRegressor,
    compute_close_day_phase2,
    estimate_lambda,
    fit_lambda_regressor,
    load_default_regressor,
)
from rotten_tomatoes_forecasting.p_fresh import estimate_p_fresh

__version__ = "0.2.0"

__all__ = [
    "compute_edge",
    "estimate_lambda",
    "estimate_p_fresh",
    "fit_lambda_regressor",
    "load_default_regressor",
    "extract_lambda_features",
    "compute_close_day_phase2",
    "LambdaRegressor",
    "LambdaPrediction",
    "naive_lambda",
    "naive_p_fresh",
    "__version__",
]
