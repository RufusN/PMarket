import pandas as pd
from statsmodels.tsa.stattools import coint
from stationarity_checks import check_stationarity
import argparse
import os
import matplotlib.pyplot as plt
import statsmodels.api as sm


def load_time_series(parquet_path):
    """
    Load time series data from a Parquet file.
    Returns a DataFrame where each column corresponds to a token's time series,
    and the columns are named <market_id>_<token_id>.
    """
    print(f"Loading Parquet file from: {parquet_path}")
    try:
        df = pd.read_parquet(parquet_path)
    except Exception as e:
        print(f"Error reading Parquet file: {e}")
        return None, None

    # Reorganize the data into a single DataFrame where each token is a column
    combined_data = {}
    market_metadata = {}
    for idx, row in df.iterrows():
        market_id = row["market_id"]
        market_name = row["market_name"]
        token_id = row["token_id"]
        timestamps = row["timestamps"]
        prices = row["prices"]

        # Create a Pandas Series with prices indexed by timestamps
        try:
            series = pd.Series(data=prices, index=pd.to_datetime(timestamps, unit='s'))
        except Exception as e:
            print(f"Error processing time series for token {token_id}: {e}")
            continue

        # Combine series into a DataFrame
        column_name = f"{market_id}_{token_id}"  # Create unique column name
        combined_data[column_name] = series

        # Store market metadata for each column
        market_metadata[column_name] = {
            "market_id": market_id,
            "market_name": market_name,
            "token_id": token_id
        }

    # Convert the combined data into a single DataFrame
    combined_df = pd.DataFrame(combined_data)
    print(f"Loaded {len(combined_data)} tokens into DataFrame.")
    return combined_df, market_metadata


def filter_and_calculate_correlations_masked(data, market_metadata, corr_threshold=0.8, pval_threshold=0.05):
    """
    Compute correlations and perform cointegration tests for pairs of tokens from different markets.
    """
    # 1) Compute the full correlation matrix
    corr_matrix = data.corr()

    # 2) Build a dictionary mapping column names to market IDs
    col_to_market = {col: col.split("_")[0] for col in data.columns}

    # ----------------------------------------------------
    # Find all cross-market pairs with correlation >= corr_threshold
    # ----------------------------------------------------
    high_corr_pairs = []
    for i in corr_matrix.columns:
        for j in corr_matrix.columns:
            if i == j or col_to_market[i] == col_to_market[j]:
                continue  # Skip same-token and same-market pairs

            current_corr = corr_matrix.loc[i, j]
            if i < j and current_corr >= corr_threshold:
                high_corr_pairs.append((i, j, current_corr))

    # Sort pairs by correlation
    high_corr_pairs.sort(key=lambda x: x[2], reverse=True)
    print(f"\nFound {len(high_corr_pairs)} cross-market pairs with correlation >= {corr_threshold}.\n")

    # ---------------------------------------------------------
    # Run stationarity checks & cointegration tests
    # ---------------------------------------------------------
    coint_results = []
    for (tokA, tokB, corr_val) in high_corr_pairs:
        # Drop NaN and align
        seriesA = data[tokA].dropna()
        seriesB = data[tokB].dropna()
        aligned = pd.concat([seriesA, seriesB], axis=1).dropna()
        alignedA = aligned[tokA]
        alignedB = aligned[tokB]

        # Stationarity checks for each series
        stationarityA = check_stationarity(seriesA)
        stationarityB = check_stationarity(seriesB)

        if stationarityA is None or stationarityB is None:
            # One or both stationarity checks might have failed
            continue

        # Perform cointegration test
        try:
            test_stat, pvalue, crit_values = coint(alignedA, alignedB)
        except Exception as e:
            print(f"Cointegration test failed for {tokA} & {tokB}: {e}")
            continue

        # Save results
        coint_result = {
            "token_1": tokA,
            "market_1_id": market_metadata[tokA]["market_id"],
            "market_1_name": market_metadata[tokA]["market_name"],
            "token_2": tokB,
            "market_2_id": market_metadata[tokB]["market_id"],
            "market_2_name": market_metadata[tokB]["market_name"],
            "corr": corr_val,
            "test_stat": test_stat,
            "pvalue": pvalue,
            "crit_values": crit_values,
            "cointegrated": (pvalue < pval_threshold)
        }
        coint_results.append(coint_result)

    # Convert cointegration results to a DataFrame
    coint_df = pd.DataFrame(coint_results).sort_values("pvalue", ascending=True)

    # Print cointegrated pairs
    print(f"\nCointegration results (p-value < {pval_threshold}):")
    cointegrated_pairs = coint_df[coint_df["cointegrated"]]
    for _, row in cointegrated_pairs.iterrows():
        print(
            f"{row['token_1']} & {row['token_2']} | Corr={row['corr']:.3f} | "
            f"p-value={row['pvalue']:.4f} -> COINTEGRATED"
        )

    return {
        "coint_results": coint_df
    }


def plot_cointegrated_residuals(data, coint_df):
    """
    For each cointegrated pair, run a simple OLS regression to get the residuals
    and then plot the residuals vs. time.
    """
    # Filter only cointegrated pairs
    cointegrated_df = coint_df[coint_df["cointegrated"]]

    if cointegrated_df.empty:
        print("No cointegrated pairs found. Skipping residual plots.")
        return

    for _, row in cointegrated_df.iterrows():
        tokA = row["token_1"]
        tokB = row["token_2"]
        nameA = f"{row['market_1_name']}"
        nameB = f"{row['market_2_name']}"

        # Align the two series
        seriesA = data[tokA].dropna()
        seriesB = data[tokB].dropna()
        aligned = pd.concat([seriesA, seriesB], axis=1).dropna()
        alignedA = aligned[tokA]
        alignedB = aligned[tokB]

        # Run a simple OLS: A ~ B
        # We add a constant to B so that the model is A = alpha + beta*B + e
        model = sm.OLS(alignedA, sm.add_constant(alignedB)).fit()
        residuals = model.resid

        # Plot the residuals
        plt.figure(figsize=(10, 5))
        residuals.plot()
        plt.axhline(0, color='red', linestyle='--')
        plt.title(f"Residuals for {nameA} vs. {nameB}")
        plt.xlabel("Time")
        plt.ylabel("Residual")
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform correlation and cointegration analysis on a Parquet file.")
    parser.add_argument("--input", required=True, help="Path to the input Parquet file.")
    parser.add_argument("--corr_threshold", type=float, default=0.9, help="Correlation threshold (default: 0.8).")
    parser.add_argument("--pval_threshold", type=float, default=0.01, help="P-value threshold (default: 0.01).")

    args = parser.parse_args()

    # Load the data
    combined_data, market_metadata = load_time_series(args.input)

    if combined_data is not None and not combined_data.empty:
        # Perform correlation and cointegration analysis
        result = filter_and_calculate_correlations_masked(
            combined_data,
            market_metadata,
            corr_threshold=args.corr_threshold,
            pval_threshold=args.pval_threshold
        )

        # Save results to a Parquet file in the same folder as the input
        output_file = os.path.join(os.path.dirname(args.input), "cointegration_results.parquet")
        result["coint_results"].to_parquet(output_file, index=False)
        print(f"Cointegration results saved to '{output_file}'.")

        # Plot residuals for cointegrated pairs
        plot_cointegrated_residuals(combined_data, result["coint_results"])
    else:
        print("No valid data was loaded. Ensure the Parquet file is correctly formatted.")
