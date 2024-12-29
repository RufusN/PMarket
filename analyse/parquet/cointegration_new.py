#!/usr/bin/env python3

import argparse
import pandas as pd
import os
import numpy as np
from statsmodels.tsa.stattools import coint
import warnings
# Import your stationarity check function
# Ensure this module is accessible or adjust the import as needed
from stationarity_checks import check_stationarity

# Define a minimum sample size for ADF test
# MIN_SAMPLE_SIZE = 20

def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze time-series data from a Parquet file for correlations and cointegration."
    )
    parser.add_argument(
        "--parquet",
        required=True,
        help="Path to the aggregated Parquet file."
    )
    parser.add_argument(
        "--corr_threshold",
        type=float,
        default=0.8,
        help="Correlation threshold for filtering token pairs (default: 0.8)."
    )
    parser.add_argument(
        "--pval_threshold",
        type=float,
        default=0.05,
        help="P-value threshold for cointegration test (default: 0.05)."
    )
    parser.add_argument(
        "--coint_parquet",
        type=str,
        default="cointegration_results.parquet",
        help="Path to save the cointegration results as a Parquet file (default: 'cointegration_results.parquet')."
    )
    parser.add_argument(
        "--coint_csv",
        type=str,
        default="cointegration_results.csv",
        help="Path to save the cointegration results as a CSV file (default: 'cointegration_results.csv')."
    )
    return parser.parse_args()

def load_time_series(parquet_path):
    """
    Read the aggregated Parquet file and organize time series data.
    Returns a dictionary mapping token_id to a dictionary containing market_id, market_name, and pandas Series.
    """
    print(f"\nReading Parquet file: {parquet_path}")
    try:
        df_agg = pd.read_parquet(parquet_path)
    except Exception as e:
        print(f"Error reading Parquet file: {e}")
        return {}

    token_series = {}
    for idx, row in df_agg.iterrows():
        market_id = row.get("market_id")
        market_name = row.get("market_name")  # Extract market_name
        token_id = row.get("token_id")
        timestamps = row.get("timestamp", [])  # list or array
        prices = row.get("price", [])          # list or array

        # Validate data types
        if not isinstance(timestamps, (list, tuple, pd.Series, np.ndarray)):
            print(f"Row {idx}: 'timestamp' is not a list-like object - skipping.")
            continue
        if not isinstance(prices, (list, tuple, pd.Series, np.ndarray)):
            print(f"Row {idx}: 'price' is not a list-like object - skipping.")
            continue
        if len(timestamps) != len(prices):
            print(f"Row {idx}: 'timestamp' and 'price' lengths do not match - skipping.")
            continue

        # Create a pandas Series for the price, indexed by timestamp
        try:
            # Convert Unix timestamps to pandas datetime
            series = pd.Series(data=prices, index=pd.to_datetime(timestamps, unit='s'))
        except Exception as e:
            print(f"Row {idx}: Error converting timestamps to datetime - {e} - skipping.")
            continue

        # # Check if the series has sufficient data points
        # if len(series) < MIN_SAMPLE_SIZE:
        #     print(f"Row {idx}: Token {token_id} has insufficient data points ({len(series)}). Expected: {MIN_SAMPLE_SIZE} - skipping.")
        #     continue

        # Check for zero variance
        if series.std() == 0:
            print(f"Row {idx}: Token {token_id} has zero variance - skipping.")
            continue

        token_series[token_id] = {
            "market_id": market_id,
            "market_name": market_name,  # Include market_name
            "series": series
        }

    print(f"Loaded {len(token_series)} tokens with valid time series data.")
    return token_series

def find_cointegrated_pairs(token_series, corr_threshold=0.8, pval_threshold=0.05):
    """
    Iterate over all unique pairs of tokens from different markets,
    compute correlation, and perform cointegration tests.
    Returns a list of dictionaries with cointegration results.
    Only includes pairs that are cointegrated (pvalue < pval_threshold).
    """
    from itertools import combinations

    coint_results = []
    tokens = list(token_series.keys())

    # Generate all unique pairs (unordered) using combinations
    for tokA, tokB in combinations(tokens, 2):
        marketA = token_series[tokA]["market_id"]
        marketB = token_series[tokB]["market_id"]

        # Skip pairs from the same market
        if marketA == marketB:
            continue

        seriesA = token_series[tokA]["series"]
        seriesB = token_series[tokB]["series"]

        # Align the two series on their timestamps (intersection)
        aligned = pd.concat([seriesA, seriesB], axis=1).dropna()
        # if aligned.empty:
        #     print(f"No overlapping timestamps for {tokA} & {tokB} - skipping.")
        #     continue

        alignedA = aligned.iloc[:, 0]
        alignedB = aligned.iloc[:, 1]

        # Compute Pearson correlation
        stdA = alignedA.std()
        stdB = alignedB.std()

        corr = alignedA.corr(alignedB)
        if pd.isna(corr):
            print(f"Correlation between {tokA} & {tokB} is NaN - skipping.")
            continue

        if corr < corr_threshold:
            continue  # Skip pairs below the correlation threshold

        # Perform stationarity checks
        stationarityA = check_stationarity(alignedA, alpha=0.05)
        stationarityB = check_stationarity(alignedB, alpha=0.05)

        # For cointegration, we generally want series that are I(1), i.e., NOT stationary by ADF test
        if stationarityA.get("ADF_Stationary", True):
            # print(f"Skipping cointegration: {tokA} appears stationary (I(0)) by ADF test.")
            continue
        if stationarityB.get("ADF_Stationary", True):
            # print(f"Skipping cointegration: {tokB} appears stationary (I(0)) by ADF test.")
            continue

        # Perform Engle-Granger cointegration test
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # Suppress warnings from coint
                test_stat, pvalue, crit_values = coint(alignedA, alignedB)
        except ValueError as ve:
            print(f"Cointegration test failed for {tokA} & {tokB}: {ve}")
            continue

        # Only include pairs that are cointegrated
        if pvalue < pval_threshold:
            coint_result = {
                "token_1": tokA,
                "market_1_id": marketA,        # Market ID for token_1
                "market_1_name": token_series[tokA]["market_name"],  # Market Name for token_1
                "token_2": tokB,
                "market_2_id": marketB,        # Market ID for token_2
                "market_2_name": token_series[tokB]["market_name"],  # Market Name for token_2
                "corr": corr,
                "test_stat": test_stat,
                "pvalue": pvalue,
                "crit_values_1%": crit_values[0],
                "crit_values_5%": crit_values[1],
                "crit_values_10%": crit_values[2],
                "cointegrated": True  # Since pvalue < pval_threshold
            }
            coint_results.append(coint_result)
            # print(f"Cointegrated pair found: {tokA} ({coint_result['market_1_name']}) & {tokB} ({coint_result['market_2_name']}) with p-value {pvalue:.5f}")

    print(f"\nFound {len(coint_results)} cross-market cointegrated pairs with correlation >= {corr_threshold} and passed stationarity checks.")
    return coint_results

def main():
    args = parse_args()

    parquet_path = args.parquet

    if not os.path.exists(parquet_path):
        print(f"Error: File '{parquet_path}' does not exist.")
        os.sys.exit(1)

    # Load the time series data
    token_series = load_time_series(parquet_path)

    if not token_series:
        print("No valid token time series data found. Exiting.")
        os.sys.exit(1)

    # Find cointegrated pairs
    coint_results = find_cointegrated_pairs(
        token_series,
        corr_threshold=args.corr_threshold,
        pval_threshold=args.pval_threshold
    )

    # Convert results to DataFrame
    if coint_results:
        coint_df = pd.DataFrame(coint_results).sort_values("pvalue", ascending=True)
        print("\nCointegration Test Results:")
        print(coint_df)

        # Save to CSV
        try:
            coint_df.to_csv(args.coint_csv, index=False)
            print(f"\nCointegration results saved to '{args.coint_csv}'.")
        except Exception as e:
            print(f"Error saving CSV file: {e}")

        # Save to Parquet
        try:
            coint_df.to_parquet(args.coint_parquet, index=False)
            print(f"Cointegration results also saved to Parquet file '{args.coint_parquet}'.")
        except Exception as e:
            print(f"Error saving Parquet file: {e}")
    else:
        print("\nNo cointegrated pairs found to save.")

if __name__ == "__main__":
    main()
