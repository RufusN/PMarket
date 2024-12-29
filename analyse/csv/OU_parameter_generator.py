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
    #   kappa = beta_hat / dt
    #   theta = alpha_hat / (kappa * dt)
    kappa = -beta_hat / dt  # note we often have -beta_hat if we wrote dS = alpha - beta*S
    if kappa == 0:
        # Edge case: if there's no reversion
        return 0, 0, 0

    theta = alpha_hat / (kappa * dt)

    # Estimate sigma from residual variance:
    # dS - (alpha_hat - beta_hat * S_t)
    residuals = Y - (alpha_hat + beta_hat * X)
    # In discrete form, sigma * sqrt(dt) is stdev of residuals
    sigma_hat = np.std(residuals, ddof=1) / np.sqrt(dt)

    # Make sure kappa is > 0 if we truly have mean reversion
    # If you prefer the sign in the regression differently, you may adapt
    # For standard OU, we expect kappa > 0
    # Otherwise we might interpret it as negative speed or no mean reversion
    return kappa, theta, sigma_hat


# --------------------------------------------------
# (B) MAIN SCRIPT to run OU for cointegrated pairs
# --------------------------------------------------
def run_ou_on_cointegrated_pairs(
    cointegration_csv="cointegration_results.csv",
    time_series_dir="/Users/ru/Polymarket/time_series",
    output_csv="ou_parameters.csv"
):
    """
    1. Read cointegration results
    2. For each row where 'cointegrated' == True, read the two CSVs
    3. Align data, form spread S = X - Y
    4. Fit OU model => (kappa, theta, sigma)
    5. Save results to output_csv
    """

    # Load the cointegration results
    coint_df = pd.read_csv(cointegration_csv)

    # Filter to only the pairs that are cointegrated
    coint_df = coint_df[coint_df["cointegrated"] == True]

    ou_results = []

    for _, row in coint_df.iterrows():
        token1 = row["token_1"]
        token2 = row["token_2"]
        correlation = row["corr"]
        pvalue = row["pvalue"]

        print(f"\nProcessing pair: {token1} & {token2} | p-value={pvalue:.4f}")

        # Build file paths (assuming CSV names match exactly the token identifiers + ".csv")
        file1 = os.path.join(time_series_dir, f"{token1}.csv")
        file2 = os.path.join(time_series_dir, f"{token2}.csv")

        # Check if files exist
        if not os.path.isfile(file1) or not os.path.isfile(file2):
            print(f"WARNING: Missing CSV for {token1} or {token2}. Skipping.")
            continue

        # Read CSVs
        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)

        # We assume each has a 'timestamp' and maybe a 'price' column
        # Let's rename them consistently and merge
        df1 = df1.rename(columns={"price": "price1", "timestamp": "timestamp"})
        df2 = df2.rename(columns={"price": "price2", "timestamp": "timestamp"})

        # Merge on timestamp (inner join)
        merged = pd.merge(df1, df2, on="timestamp", how="inner")
        merged = merged.sort_values("timestamp")

        # S = X - Y (you could refine by including the cointegration ratio if known)
        S = merged["price1"] - merged["price2"]

        # Fit OU
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

        print(f"  OU Fit => kappa={kappa:.4f}, theta={theta:.4f}, sigma={sigma:.4f}")

    # Save all OU parameter results to a new CSV
    ou_df = pd.DataFrame(ou_results)
    ou_df.to_csv(output_csv, index=False)
    print(f"\nOrnstein–Uhlenbeck parameters saved to '{output_csv}'")


if __name__ == "__main__":
    # Example usage:
    run_ou_on_cointegrated_pairs(
        cointegration_csv="cointegration_results.csv",
        time_series_dir="/Users/ru/Polymarket/time_series",
        output_csv="ou_parameters.csv"
    )
