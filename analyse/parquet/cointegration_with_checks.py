import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
from load_time_series import market_data
from stationarity_checks import check_stationarity

# For cointegration
from statsmodels.tsa.stattools import coint

def filter_and_calculate_correlations_masked(data, corr_threshold=0.8, pval_threshold=0.05):
    """
    1) Compute the correlation matrix for all columns/tokens.
    2) Build a mask that is True where tokens share the same market (i.e., same prefix).
    3) (Optionally) plot a heatmap, using the mask to hide same-market cells.
    4) Return a dataframe of cointegration results for pairs whose correlation is above `corr_threshold`
       and where both series appear to be non-stationary (likely I(1)), then pass an Engle–Granger test.
    """

    # 1) Compute the full correlation matrix
    corr_matrix = data.corr()

    # 2) Build a dictionary from column -> market_id
    col_to_market = {col: col.split("_")[0] for col in data.columns}

    # 3) Create a boolean mask (DataFrame) to mark same-market pairs
    mask = pd.DataFrame(False, index=corr_matrix.index, columns=corr_matrix.columns)
    for c1 in corr_matrix.columns:
        for c2 in corr_matrix.columns:
            # If both tokens belong to the same market, we mask (set True) to exclude them
            if col_to_market[c1] == col_to_market[c2]:
                mask.loc[c1, c2] = True

    # # 4) (OPTIONAL) Plot the heatmap using seaborn, masking the same-market cells
    # plt.figure(figsize=(12, 8))
    # sns.heatmap(
    #     corr_matrix,
    #     mask=mask,         # hide same-market correlations
    #     cmap="coolwarm",
    #     center=0,
    #     annot=False,       # set True if you want numeric values in cells
    #     fmt=".2f",
    #     cbar=True,
    #     square=False,      # set True if you'd prefer square cells
    #     linewidths=0.5,
    #     linecolor='gray'
    # )
    # plt.title("Cross-Market Token Correlations (Masked Same-Market Pairs)")
    # plt.xticks(rotation=45, ha="right")
    # plt.yticks(rotation=0)
    # plt.tight_layout()
    # plt.show()

    # ----------------------------------------------------
    # Find all cross-market pairs with correlation >= corr_threshold
    # ----------------------------------------------------
    high_corr_pairs = []
    for i in corr_matrix.columns:
        for j in corr_matrix.columns:
            # skip same-market or same token
            if i == j or col_to_market[i] == col_to_market[j]:
                continue

            current_corr = corr_matrix.loc[i, j]
            # Only process one triangle (i < j) to avoid duplicates
            if i < j and current_corr >= corr_threshold:
                high_corr_pairs.append((i, j, current_corr))

    # Sort them descending by correlation
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

        # ---- Stationarity checks for each series ----
        stationarityA = check_stationarity(alignedA, alpha=0.05)
        stationarityB = check_stationarity(alignedB, alpha=0.05)

        # If ADF_Stationary == True => the series is likely I(0).
        # For cointegration, we generally want series that are I(1), i.e. NOT stationary by ADF test.
        # So if either is stationary, skip cointegration test.
        if stationarityA["ADF_Stationary"]:
            print(f"Skipping cointegration: {tokA} appears stationary (I(0)) by ADF test.")
            continue
        if stationarityB["ADF_Stationary"]:
            print(f"Skipping cointegration: {tokB} appears stationary (I(0)) by ADF test.")
            continue

        # ---- Proceed with Engle–Granger cointegration if both are non-stationary (likely I(1)) ----
        test_stat, pvalue, crit_values = coint(alignedA, alignedB)

        coint_result = {
            "token_1": tokA,
            "token_2": tokB,
            "corr": corr_val,
            "test_stat": test_stat,
            "pvalue": pvalue,
            "crit_values": crit_values,
            "cointegrated": (pvalue < pval_threshold)
        }
        coint_results.append(coint_result)

    # Convert all cointegration results to a DataFrame, whether cointegrated or not
    coint_df = pd.DataFrame(coint_results).sort_values("pvalue", ascending=True)

    # Print out cointegrated pairs
    print(f"\nCointegration results (p-value < {pval_threshold}):")
    cointegrated_pairs = coint_df[coint_df["cointegrated"] == True]
    for _, row in cointegrated_pairs.iterrows():
        print(
            f"{row['token_1']} & {row['token_2']} | Corr={row['corr']:.3f} | "
            f"p-value={row['pvalue']:.4f} -> COINTEGRATED"
        )

    return {
        "coint_results": coint_df
    }

if __name__ == "__main__":
    directory = "/Users/ru/Polymarket/time_series"
    
    # Load all market time-series into a single DataFrame
    combined_data = market_data(directory)
    
    if not combined_data.empty:
        print("Data loaded. Head of the combined DataFrame:")
        print(combined_data.head())
        
        # Calculate cross-market correlations & run cointegration analysis
        result = filter_and_calculate_correlations_masked(
            combined_data,
            corr_threshold=0.8,
            pval_threshold=0.05
        )
        
        # Save the cointegration test results to CSV
        result["coint_results"].to_csv("cointegration_results.csv", index=False)
        print("Cointegration results saved to 'cointegration_results.csv'.")

    else:
        print("No data was loaded. Ensure the directory contains valid CSV files.")
