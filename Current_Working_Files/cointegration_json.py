import pandas as pd
from statsmodels.tsa.stattools import coint
from stationarity_checks import check_stationarity
import argparse
import os
import matplotlib.pyplot as plt
import statsmodels.api as sm
import json
import numpy as np


def load_time_series_from_json(json_path):
    """
    Load time series data from a JSON file.
    Returns a DataFrame where each column corresponds to a token's time series,
    and the columns are named <market_id>_<token_id>.
    """
    print(f"Loading JSON file from: {json_path}")
    try:
        with open(json_path, 'r') as file:
            data = json.load(file)
    except Exception as e:
        print(f"Error reading JSON file: {e}")
        return None, None

    # Reorganize the data into a single DataFrame where each token is a column
    combined_data = {}
    market_metadata = {}
    for market in data:
        market_id = market["market_id"]
        market_name = market["market_name"]

        for token in market["tokens"]:
            token_id = token["token_id"]
            time_series = token["time_series"]

            # Extract timestamps and prices
            try:
                timestamps = [point["t"] for point in time_series]
                prices = [point["p"] for point in time_series]
                if prices and (prices[-1] < 0.05 or prices[-1] > 0.95):
                    print(f"Skipping token {token_id} with price {prices[-1]}")
                    continue
                series = pd.Series(data=prices, index=pd.to_datetime(timestamps, unit='s'))
            except Exception as e:
                print(f"Error processing time series for token {token_id}: {e}")
                continue

            # Combine series into a DataFrame
            column_name = f"{token_id}"  # Create unique column name
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


def filter_and_calculate_correlations_masked(data, market_metadata, corr_threshold, pval_threshold):
    """
    Compute correlations and perform cointegration tests for pairs of tokens from different markets.
    """
    corr_matrix = data.corr()
    col_to_market = {col: col.split("_")[0] for col in data.columns}

    high_corr_pairs = []
    for i in corr_matrix.columns:
        for j in corr_matrix.columns:
            if i == j or col_to_market[i] == col_to_market[j]:
                continue

            current_corr = corr_matrix.loc[i, j]
            if i < j and current_corr >= corr_threshold:
                high_corr_pairs.append((i, j, current_corr))

    high_corr_pairs.sort(key=lambda x: x[2], reverse=True)
    print(f"\nFound {len(high_corr_pairs)} cross-market pairs with correlation >= {corr_threshold}.\n")

    coint_results = []
    for (tokA, tokB, corr_val) in high_corr_pairs:
        seriesA = data[tokA].dropna()
        seriesB = data[tokB].dropna()
        aligned = pd.concat([seriesA, seriesB], axis=1).dropna()
        alignedA = aligned[tokA]
        alignedB = aligned[tokB]

        stationarityA = check_stationarity(seriesA)
        stationarityB = check_stationarity(seriesB)

        if stationarityA is None or stationarityB is None:
            continue

        try:
            test_stat, pvalue, crit_values = coint(alignedA, alignedB)
        except Exception as e:
            print(f"Cointegration test failed for {tokA} & {tokB}: {e}")
            continue

        coint_result = {
            "token_1": tokA,
            "market_1_id": market_metadata[tokA]["market_id"],
            "market_1_name": market_metadata[tokA]["market_name"],
            "token_2": tokB,
            "market_2_id": market_metadata[tokB]["market_id"],
            "market_2_name": market_metadata[tokB]["market_name"],
            "corr": corr_val,
            "test_stat": float(test_stat),  # Ensure float type for JSON
            "pvalue": float(pvalue),  # Ensure float type for JSON
            "crit_values": crit_values.tolist(),  # Convert ndarray to list
            "cointegrated": bool(pvalue < pval_threshold)  # Ensure Python bool type
        }
        coint_results.append(coint_result)

    coint_df = pd.DataFrame(coint_results).sort_values("pvalue", ascending=True)

    print(f"\nCointegration results (p-value < {pval_threshold}):")
    cointegrated_pairs = coint_df[coint_df["cointegrated"]]
    for _, row in cointegrated_pairs.iterrows():
        print(
            f"{row['token_1']} & {row['token_2']} | Corr={row['corr']:.3f} | "
            f"p-value={row['pvalue']:.4f} -> COINTEGRATED"
        )

    return {"coint_results": coint_df, "raw_results": coint_results}


def save_results_to_json(coint_results):
    """Save filtered cointegration results to a JSON file."""
    output_file = "cointegration_results.json"
    print(f"Saving results to JSON: {output_file}")

    try:
        # Filter results where cointegrated == True
        filtered_results = [result for result in coint_results if result["cointegrated"]]

        # Sort the filtered results by p-value in ascending order
        sorted_results = sorted(filtered_results, key=lambda x: x["pvalue"])

        # Save to JSON
        with open(output_file, 'w') as file:
            json.dump(sorted_results, file, indent=4)
        
        print(f"Results saved to {output_file}")

    except Exception as e:
        print(f"Error saving JSON file: {e}")


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
    parser = argparse.ArgumentParser(description="Perform correlation and cointegration analysis on a JSON file.")
    parser.add_argument("--input", required=True, help="Path to the input JSON file.")
    parser.add_argument("--corr_threshold", type=float, default=0.9, help="Correlation threshold (default: 0.9).")
    parser.add_argument("--pval_threshold", type=float, default=0.01, help="P-value threshold (default: 0.01).")

    args = parser.parse_args()

    combined_data, market_metadata = load_time_series_from_json(args.input)

    if combined_data is not None and not combined_data.empty:
        result = filter_and_calculate_correlations_masked(
            combined_data,
            market_metadata,
            corr_threshold=args.corr_threshold,
            pval_threshold=args.pval_threshold
        )

        save_results_to_json(result["raw_results"])

        # Optionally, plot residuals for cointegrated pairs
        plot_cointegrated_residuals(combined_data, result["coint_results"])
    else:
        print("No valid data was loaded. Ensure the JSON file is correctly formatted.")
