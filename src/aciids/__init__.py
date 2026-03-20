"""Public package interface for the polished ACIIDS repository."""

from .pricing import (
    OptionContract,
    benchmark_contract,
    binomial_price,
    black_scholes_price,
    monte_carlo_price,
)
from .sentiment import aggregate_daily_sentiment, chunk_text

__all__ = [
    "OptionContract",
    "aggregate_daily_sentiment",
    "benchmark_contract",
    "binomial_price",
    "black_scholes_price",
    "chunk_text",
    "monte_carlo_price",
]
