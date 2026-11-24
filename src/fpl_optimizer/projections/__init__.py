"""Expected-points projections: FPL's own, a naive baseline, and the JAX model."""

from .base import PREDICTION_COLUMNS, Projection, aggregate_to_gw, empty_predictions
from .baselines import EPProjection, NaiveProjection
from .model import JaxPointsModel

#: Strategy name -> factory, used by the CLI and the backtester.
STRATEGIES: dict[str, type] = {
    "model": JaxPointsModel,
    "ep": EPProjection,
    "naive": NaiveProjection,
}

__all__ = [
    "PREDICTION_COLUMNS",
    "Projection",
    "aggregate_to_gw",
    "empty_predictions",
    "EPProjection",
    "NaiveProjection",
    "JaxPointsModel",
    "STRATEGIES",
]
