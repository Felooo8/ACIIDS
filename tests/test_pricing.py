import math

import pytest

from aciids.pricing import (
    OptionContract,
    benchmark_contract,
    binomial_price,
    black_scholes_price,
    monte_carlo_price,
)


@pytest.fixture
def contract() -> OptionContract:
    return OptionContract(
        spot=100.0,
        strike=100.0,
        time_to_expiry=1.0,
        risk_free_rate=0.05,
        volatility=0.2,
        option_type="call",
    )


def test_black_scholes_matches_reference_value(contract: OptionContract) -> None:
    assert black_scholes_price(contract) == pytest.approx(10.4506, rel=1e-3)


def test_binomial_stays_close_to_black_scholes(contract: OptionContract) -> None:
    price = binomial_price(contract, steps=400)
    assert price == pytest.approx(black_scholes_price(contract), rel=2e-2)


def test_monte_carlo_is_reasonably_close(contract: OptionContract) -> None:
    price = monte_carlo_price(contract, simulations=200_000, seed=11)
    assert price == pytest.approx(black_scholes_price(contract), rel=3e-2)


def test_benchmark_contract_returns_expected_keys(contract: OptionContract) -> None:
    benchmark = benchmark_contract(contract)
    assert set(benchmark) == {"black_scholes", "binomial", "monte_carlo"}
    assert all(math.isfinite(value) for value in benchmark.values())


def test_invalid_option_type_raises() -> None:
    with pytest.raises(ValueError):
        black_scholes_price(
            OptionContract(
                spot=100,
                strike=100,
                time_to_expiry=1,
                risk_free_rate=0.05,
                volatility=0.2,
                option_type="straddle",
            )
        )


def test_black_scholes_at_expiry_returns_intrinsic() -> None:
    itm_call = OptionContract(spot=110, strike=100, time_to_expiry=0, risk_free_rate=0.05, volatility=0.2)
    assert black_scholes_price(itm_call) == pytest.approx(10.0)
    otm_call = OptionContract(spot=90, strike=100, time_to_expiry=0, risk_free_rate=0.05, volatility=0.2)
    assert black_scholes_price(otm_call) == pytest.approx(0.0)


def test_black_scholes_zero_volatility_returns_forward_payoff() -> None:
    # With zero vol the price is PV of the deterministic forward payoff.
    contract = OptionContract(spot=100, strike=100, time_to_expiry=1.0, risk_free_rate=0.05, volatility=0.0)
    forward = 100 * math.exp(0.05 * 1.0)
    expected = math.exp(-0.05 * 1.0) * max(forward - 100, 0.0)
    assert black_scholes_price(contract) == pytest.approx(expected)


def test_binomial_at_expiry_returns_intrinsic() -> None:
    itm_call = OptionContract(spot=110, strike=100, time_to_expiry=0, risk_free_rate=0.05, volatility=0.2)
    assert binomial_price(itm_call) == pytest.approx(10.0)


def test_monte_carlo_at_expiry_returns_intrinsic() -> None:
    itm_call = OptionContract(spot=110, strike=100, time_to_expiry=0, risk_free_rate=0.05, volatility=0.2)
    assert monte_carlo_price(itm_call) == pytest.approx(10.0)
