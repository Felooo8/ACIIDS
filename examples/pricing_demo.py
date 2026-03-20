"""Small demo script for the cleaned public API."""

from aciids.pricing import OptionContract, benchmark_contract

if __name__ == "__main__":
    contract = OptionContract(
        spot=180.0,
        strike=175.0,
        time_to_expiry=30 / 365,
        risk_free_rate=0.04,
        volatility=0.24,
        option_type="call",
    )
    for model_name, price in benchmark_contract(contract).items():
        print(f"{model_name:>14}: {price:.4f}")
