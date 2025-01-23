import pandas as pd
import numpy as np
from statsmodels.regression.linear_model import OLS
import json
import os

def load_json(filepath):
    try:
        with open(filepath, 'r') as file:
            return json.load(file)
    except Exception as e:
        print(f"ERROR: Failed to load JSON file '{filepath}'. Exception: {e}")
        return None


def get_time_series_and_market_name(market_data, token_id):
    """
    Extract time series and market name for a specific token from market data JSON.
    """
    for market in market_data:
        for token in market["tokens"]:
            if token["token_id"] == token_id:
                time_series = token.get("time_series", [])
                timestamps = [point["t"] for point in time_series]
                prices = [point["p"] for point in time_series]
                return pd.Series(data=prices, index=pd.to_datetime(timestamps, unit='s')), market["market_name"]
    return pd.Series(dtype=float), None


def fit_ornstein_uhlenbeck(spread_series, dt=1.0):
    """
    Fit Ornstein–Uhlenbeck parameters: kappa, theta, sigma.
    """
    spread_series = spread_series.dropna()
    dS = spread_series.diff().dropna()
    S_lag = spread_series.shift(1).dropna()

    valid_idx = dS.index.intersection(S_lag.index)
    dS = dS.loc[valid_idx]
    S_lag = S_lag.loc[valid_idx]

    X = S_lag.values
    Y = dS.values
    X_with_const = np.column_stack((np.ones_like(X), X))

    model = OLS(Y, X_with_const).fit()
    alpha_hat, beta_hat = model.params

    kappa = -beta_hat / dt
    if kappa == 0:
        return 0, 0, 0

    theta = alpha_hat / (kappa * dt)
    residuals = Y - (alpha_hat + beta_hat * X)
    sigma_hat = np.std(residuals, ddof=1) / np.sqrt(dt)

    return kappa, theta, sigma_hat


def run_ou_on_cointegrated_pairs(
    cointegration_json="cointegration_results.json",
    market_data_json="markets_with_tokens_time_series.json",
    output_json="ou_parameters.json"
):
    """
    Fit Ornstein–Uhlenbeck parameters for cointegrated pairs and save results.
    """
    # Load data
    coint_data = load_json(cointegration_json)
    market_data = load_json(market_data_json)

    if not coint_data or not market_data:
        print("ERROR: Failed to load input data. Exiting.")
        return

    # Filter cointegrated pairs
    coint_pairs = [pair for pair in coint_data if pair.get("cointegrated")]

    if not coint_pairs:
        print("No cointegrated pairs found. Exiting.")
        return

    ou_results = []
    total_pairs = len(coint_pairs)

    for idx, pair in enumerate(coint_pairs):
        token1 = pair["token_1"]
        token2 = pair["token_2"]
        correlation = pair.get("corr", np.nan)
        pvalue = pair.get("pvalue", np.nan)

        print(f"\nProcessing pair {idx + 1}/{total_pairs}: {token1} & {token2} | p-value={pvalue:.6f}")

        # Get time series and market names for both tokens
        series1, market_name1 = get_time_series_and_market_name(market_data, token1)
        series2, market_name2 = get_time_series_and_market_name(market_data, token2)

        if series1.empty or series2.empty:
            print(f"  WARNING: Time series data missing for {token1} or {token2}. Skipping.")
            continue

        # Align time series and compute spread
        merged = pd.concat([series1, series2], axis=1, keys=["price1", "price2"]).dropna()

        if merged.empty or len(merged) < 2:
            print(f"  WARNING: Insufficient overlapping data for {token1} and {token2}. Skipping.")
            continue

        spread = merged["price1"] - merged["price2"]

        # Fit OU model
        kappa, theta, sigma = fit_ornstein_uhlenbeck(spread)

        ou_results.append({
            "token_1": token1,
            "market_1_name": market_name1,
            "token_2": token2,
            "market_2_name": market_name2,
            "corr": correlation,
            "pvalue": pvalue,
            "kappa": kappa,
            "theta": theta,
            "sigma": sigma
        })

        print(f"  OU Fit => kappa={kappa:.6f}, theta={theta:.6f}, sigma={sigma:.6f}")

    if not ou_results:
        print("No OU parameters were computed. Exiting.")
        return

    # Save OU parameters to JSON
    try:
        with open(output_json, 'w') as file:
            json.dump(ou_results, file, indent=4)
        print(f"\nOrnstein–Uhlenbeck parameters saved to '{output_json}'")
    except Exception as e:
        print(f"ERROR: Failed to save OU parameters to '{output_json}'. Exception: {e}")


if __name__ == "__main__":
    run_ou_on_cointegrated_pairs(
        cointegration_json="cointegration_results.json",
        market_data_json="sorted_markets_with_tokens.json",
        output_json="ou_parameters.json"
    )
