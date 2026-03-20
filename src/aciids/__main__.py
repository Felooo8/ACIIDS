"""Simple command-line demo for the cleaned public package."""

from .pricing import OptionContract, benchmark_contract


def main() -> None:
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


if __name__ == "__main__":
    main()
