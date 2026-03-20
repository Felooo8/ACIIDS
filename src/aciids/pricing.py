"""Reusable option-pricing utilities extracted from the research codebase."""

from __future__ import annotations

from dataclasses import dataclass
from math import erf, exp, log, sqrt
from random import Random

_MIN_POSITIVE = 1e-8


@dataclass(frozen=True)
class OptionContract:
    """Normalized option contract parameters used across pricing models."""

    spot: float
    strike: float
    time_to_expiry: float
    risk_free_rate: float
    volatility: float
    option_type: str = "call"

    def validate(self) -> None:
        if self.spot <= 0:
            raise ValueError("spot price must be positive")
        if self.strike <= 0:
            raise ValueError("strike price must be positive")
        if self.time_to_expiry < 0:
            raise ValueError("time_to_expiry must be non-negative")
        if self.volatility < 0:
            raise ValueError("volatility must be non-negative")
        if self.option_type not in {"call", "put"}:
            raise ValueError("option_type must be either 'call' or 'put'")

    @property
    def safe_time_to_expiry(self) -> float:
        return max(self.time_to_expiry, _MIN_POSITIVE)

    @property
    def safe_volatility(self) -> float:
        return max(self.volatility, _MIN_POSITIVE)


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + erf(value / sqrt(2.0)))


def _payoff(option_type: str, terminal_price: float, strike: float) -> float:
    if option_type == "call":
        return max(terminal_price - strike, 0.0)
    return max(strike - terminal_price, 0.0)


def black_scholes_price(contract: OptionContract) -> float:
    """Return the analytical Black-Scholes price for a European option."""
    contract.validate()
    sigma = contract.safe_volatility
    time_to_expiry = contract.safe_time_to_expiry
    d1 = (
        log(contract.spot / contract.strike)
        + (contract.risk_free_rate + 0.5 * sigma**2) * time_to_expiry
    ) / (sigma * sqrt(time_to_expiry))
    d2 = d1 - sigma * sqrt(time_to_expiry)

    if contract.option_type == "call":
        return (
            contract.spot * _normal_cdf(d1)
            - contract.strike * exp(-contract.risk_free_rate * time_to_expiry) * _normal_cdf(d2)
        )

    return (
        contract.strike * exp(-contract.risk_free_rate * time_to_expiry) * _normal_cdf(-d2)
        - contract.spot * _normal_cdf(-d1)
    )


def binomial_price(contract: OptionContract, steps: int = 200) -> float:
    """Price an option with a Cox-Ross-Rubinstein binomial tree."""
    contract.validate()
    if steps <= 0:
        raise ValueError("steps must be positive")

    time_to_expiry = contract.safe_time_to_expiry
    sigma = contract.safe_volatility
    dt = time_to_expiry / steps
    up = exp(sigma * sqrt(dt))
    down = 1 / up
    growth = exp(contract.risk_free_rate * dt)
    probability = (growth - down) / (up - down)

    if not 0 <= probability <= 1:
        raise ValueError("risk-neutral probability is outside [0, 1]")

    option_values = [
        _payoff(contract.option_type, contract.spot * down ** (steps - i) * up**i, contract.strike)
        for i in range(steps + 1)
    ]

    discount = exp(-contract.risk_free_rate * dt)
    for step in range(steps, 0, -1):
        option_values = [
            discount * (probability * option_values[index + 1] + (1 - probability) * option_values[index])
            for index in range(step)
        ]

    return option_values[0]


def monte_carlo_price(
    contract: OptionContract,
    simulations: int = 10_000,
    seed: int | None = 7,
) -> float:
    """Estimate an option price with Monte Carlo simulation."""
    contract.validate()
    if simulations <= 0:
        raise ValueError("simulations must be positive")

    rng = Random(seed)
    time_to_expiry = contract.safe_time_to_expiry
    sigma = contract.safe_volatility
    drift = (contract.risk_free_rate - 0.5 * sigma**2) * time_to_expiry
    diffusion = sigma * sqrt(time_to_expiry)

    payoff_total = 0.0
    for _ in range(simulations):
        shock = rng.gauss(0.0, 1.0)
        terminal_price = contract.spot * exp(drift + diffusion * shock)
        payoff_total += _payoff(contract.option_type, terminal_price, contract.strike)

    discounted_mean_payoff = payoff_total / simulations
    return exp(-contract.risk_free_rate * time_to_expiry) * discounted_mean_payoff


def benchmark_contract(contract: OptionContract) -> dict[str, float]:
    """Return a side-by-side benchmark for the main supported pricing models."""
    return {
        "black_scholes": black_scholes_price(contract),
        "binomial": binomial_price(contract),
        "monte_carlo": monte_carlo_price(contract),
    }
