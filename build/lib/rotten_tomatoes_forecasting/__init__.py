"""rotten-tomatoes-forecasting: Poisson-binomial forecasting for Kalshi RT Tomatometer markets."""

from rotten_tomatoes_forecasting.edge import compute_edge
from rotten_tomatoes_forecasting.critic_model import (
    CriticProfiles,
    KDELambdaModel,
    build_critic_profiles,
    build_kde_lambda_model,
    estimate_lambda,
    estimate_p_fresh,
    default_training_slugs,
)

__version__ = "0.1.0"
