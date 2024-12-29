import os
import pandas as pd
import numpy as np
from statsmodels.regression.linear_model import OLS

# --------------------------------------------------
# (A) HELPER: Fit Ornstein–Uhlenbeck to a spread
# --------------------------------------------------
def fit_ornstein_uhlenbeck(spread_series, dt=1.0):
    """
    Estimate Ornstein–Uhlenbeck parameters (kappa, theta, sigma)
    using a discrete-time approximation and linear regression approach.

    Model (discretized):
      S_{t+1} = S_t + kappa * (theta - S_t) * dt + sigma * sqrt(dt) * eps
    or rearranged:
      S_{t+1} - S_t = kappa*dt*theta - kappa*dt*S_t + sigma*sqrt(dt)*eps

    We'll do a linear regression on:
      dS = alpha - beta * S_t
    where alpha = kappa*theta*dt, and beta = kappa*dt.
    Then solve for kappa, theta, sigma.

    :param spread_series: pd.Series of spread data
    :param dt: time-step size (1.0 if daily or 1.0 for each row)
    :return: (kappa, theta, sigma)
    """
    # Drop NaNs, just in case
    spread_series = spread_series.dropna()

    # Calculate dS
    dS = spread_series.diff().dropna()
    S_lag = spread_series.shift(1).dropna()

    # Align dS and S_lag
    # (They should already align by index, but let's be explicit)
    valid_idx = dS.index.intersection(S_lag.index)
    dS = dS.loc[valid_idx]
    S_lag = S_lag.loc[valid_idx]

    # Regression: dS = alpha - beta * S_{t}
    X = S_lag.values
    Y = dS.values

    # Add constant to X for the intercept
    X_with_const = np.column_stack((np.ones_like(X), X))

    # Run OLS
    model = OLS(Y, X_with_const).fit()
    alpha_hat, beta_hat = model.params

    # From alpha_hat = kappa * theta * dt, beta_hat = kappa * dt
    # Solve:
    #   kappa = -beta_hat / dt
    #   theta = alpha_hat / (kappa * dt)
    kappa = -beta_hat / dt  # note the negative sign due to model specification
    if kappa == 0:
        # Edge case: if there's no reversion
        return 0, 0, 0

    theta = alpha_hat / (kappa * dt)

    # Estimate sigma from residual variance:
    # dS - (alpha_hat - beta_hat * S_t)
    residuals = Y - (alpha_hat + beta_hat * X)
    # In discrete form, sigma * sqrt(dt) is stdev of residuals
    sigma_hat = np.std(residuals, ddof=1) / np.sqrt(dt)

    # Ensure kappa is positive for mean reversion
    return kappa, theta, sigma_hat

# --------------------------------------------------
# (B) MAIN SCRIPT to run OU for cointegrated pairs
# --------------------------------------------------
def run_ou_on_cointegrated_pairs(
    cointegration_parquet="/Users/ru/Polymarket/analyse/parquet/cointegration_results.parquet",  # Updated Path
    time_series_dir="/Users/ru/Polymarket/time_series",     # Ensure this is correct
    output_parquet="/Users/ru/Polymarket/analyse/parquet/ou_parameters.parquet"  # Updated Output Path
):
    """
    1. Read cointegration results from a Parquet file
    2. For each row where 'cointegrated' == True, read the two Parquet files
    3. Align data, form spread S = X - Y
    4. Fit OU model => (kappa, theta, sigma)
    5. Save results to output_parquet

    :param cointegration_parquet: Path to the cointegration results Parquet file
    :param time_series_dir: Directory containing time series Parquet files for each token
    :param output_parquet: Path to save the OU parameters as a Parquet file
    """
    # Load the cointegration results from Parquet
    try:
        coint_df = pd.read_parquet(cointegration_parquet)
        print(f"Loaded cointegration results from '{cointegration_parquet}'")
    except Exception as e:
        print(f"ERROR: Failed to read cointegration results from '{cointegration_parquet}'. Exception: {e}")
        return

    # Ensure 'cointegrated' column exists
    if "cointegrated" not in coint_df.columns:
        print("ERROR: 'cointegrated' column not found in the cointegration results.")
        return

    # Filter to only the pairs that are cointegrated
    coint_df = coint_df[coint_df["cointegrated"] == True]

    if coint_df.empty:
        print("No cointegrated pairs found.")
        return

    # List available Parquet files for debugging
    try:
        available_files = set(os.listdir(time_series_dir))
        print(f"Available Parquet files in '{time_series_dir}': {len(available_files)} files found.")
    except Exception as e:
        print(f"ERROR: Unable to list files in '{time_series_dir}'. Exception: {e}")
        return

    ou_results = []

    total_pairs = len(coint_df)
    for idx, row in coint_df.iterrows():
        token1 = row["token_1"]
        token2 = row["token_2"]
        correlation = row.get("corr", np.nan)
        pvalue = row.get("pvalue", np.nan)

        print(f"\nProcessing pair {idx + 1}/{total_pairs}: {token1} & {token2} | p-value={pvalue:.6f}")

        # Build file paths based on Parquet format
        file1 = f"{token1}.parquet"
        file2 = f"{token2}.parquet"

        # Check if files exist in the directory
        file1_exists = file1 in available_files
        file2_exists = file2 in available_files

        if not file1_exists or not file2_exists:
            missing_files = []
            if not file1_exists:
                missing_files.append(file1)
            if not file2_exists:
                missing_files.append(file2)
            print(f"WARNING: Missing Parquet files for {', '.join(missing_files)}. Skipping.")
            continue

        # Read time series data
        try:
            df1 = pd.read_parquet(os.path.join(time_series_dir, file1))
            df2 = pd.read_parquet(os.path.join(time_series_dir, file2))
            print(f"  Loaded time series data for {token1} and {token2}")
        except Exception as e:
            print(f"WARNING: Failed to read time series data for {token1} or {token2}. Exception: {e}. Skipping.")
            continue

        # Check for necessary columns
        if "price" not in df1.columns or "timestamp" not in df1.columns:
            print(f"WARNING: 'price' or 'timestamp' column missing in {token1}'s data. Skipping.")
            continue
        if "price" not in df2.columns or "timestamp" not in df2.columns:
            print(f"WARNING: 'price' or 'timestamp' column missing in {token2}'s data. Skipping.")
            continue

        # Rename price columns for clarity
        df1 = df1.rename(columns={"price": "price1", "timestamp": "timestamp"})
        df2 = df2.rename(columns={"price": "price2", "timestamp": "timestamp"})

        # Merge on timestamp (inner join)
        merged = pd.merge(df1, df2, on="timestamp", how="inner")
        if merged.empty:
            print(f"  WARNING: No overlapping timestamps between {token1} and {token2}. Skipping.")
            continue
        merged = merged.sort_values("timestamp")

        # Check if there are enough data points
        if len(merged) < 2:
            print(f"  WARNING: Not enough data points after merging for {token1} and {token2}. Skipping.")
            continue

        # Form spread S = price1 - price2
        S = merged["price1"] - merged["price2"]

        # Fit OU model
        kappa, theta, sigma = fit_ornstein_uhlenbeck(S)

        # Store results
        ou_results.append({
            "token_1": token1,
            "token_2": token2,
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

    # Save all OU parameter results to a Parquet file
    ou_df = pd.DataFrame(ou_results)
    try:
        ou_df.to_parquet(output_parquet, index=False)
        print(f"\nOrnstein–Uhlenbeck parameters saved to '{output_parquet}'")
    except Exception as e:
        print(f"ERROR: Failed to save OU parameters to '{output_parquet}'. Exception: {e}")

if __name__ == "__main__":
    # Example usage:
    run_ou_on_cointegrated_pairs(
        cointegration_parquet="/Users/ru/Polymarket/analyse/parquet/cointegration_results.parquet",  # Updated Path
        time_series_dir="/Users/ru/Polymarket/time_series",                                       # Ensure this is correct
        output_parquet="/Users/ru/Polymarket/analyse/parquet/ou_parameters.parquet"             # Updated Output Path
    )
