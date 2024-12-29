import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from load_time_series import market_data
from stationarity_checks import check_stationarity

# For cointegration
from statsmodels.tsa.stattools import coint

def filter_and_calculate_correlations_masked(data, corr_threshold=0.8, pval_threshold=0.05):
    """
    1) Compute the correlation matrix for all columns/tokens.
    2) Build a mask that is True where tokens share the same market (i.e., same prefix).
    3) Plot a heatmap, using the mask to hide same-market cells.
    4) Return the correlation matrix (with same-market pairs hidden in the plot),
       and run cointegration tests on pairs whose correlation is above `corr_threshold`.
    """

    # 1) Compute the full correlation matrix
    corr_matrix = data.corr()

    # 2) Build a dictionary from column -> market_id
    col_to_market = {col: col.split("_")[0] for col in data.columns}

    # 3) Create a boolean mask (DataFrame) to mark same-market pairs
    mask = pd.DataFrame(False, index=corr_matrix.index, columns=corr_matrix.columns)
    for c1 in corr_matrix.columns:
        for c2 in corr_matrix.columns:
            if col_to_market[c1] == col_to_market[c2]:
                # True indicates we want to "hide" or "mask" these same-market pairs
                mask.loc[c1, c2] = True

    # # 4) Plot the heatmap using seaborn, masking the same-market cells
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
    # # Rotate labels to prevent overlap
    # plt.xticks(rotation=45, ha="right")
    # plt.yticks(rotation=0)
    # plt.tight_layout()
    # plt.show()

    # # --------------------------------------------------
    # #  (A) Identify the single highest correlation pair
    # # --------------------------------------------------
    # masked_corr = corr_matrix.where(~mask)  # Apply mask to exclude same-market pairs
    # max_corr_pair = masked_corr.stack().idxmax()  # index of maximum correlation value
    # max_value = masked_corr.loc[max_corr_pair[0], max_corr_pair[1]]

    # # Extract details
    # market_id_1 = col_to_market[max_corr_pair[0]]
    # market_id_2 = col_to_market[max_corr_pair[1]]
    # token_1 = max_corr_pair[0]
    # token_2 = max_corr_pair[1]

    # print(f"Maximum cross-market correlation: {max_value:.3f}")
    # print(f"Between token: {token_1} (Market ID: {market_id_1})")
    # print(f"And token: {token_2} (Market ID: {market_id_2})")

    # ----------------------------------------------------
    # (B) Find all pairs with correlation above threshold
    # ----------------------------------------------------
    # We'll make a list of tuples for pairs (tokenA, tokenB, correlation).
    high_corr_pairs = []
    for i in corr_matrix.columns:
        for j in corr_matrix.columns:
            # skip same-market or same token
            if i == j or col_to_market[i] == col_to_market[j]:
                continue

            current_corr = corr_matrix.loc[i, j]
            # only process upper triangle or lower triangle once to avoid duplicates
            # e.g., i < j lexicographically
            if i < j and current_corr >= corr_threshold:
                high_corr_pairs.append((i, j, current_corr))

    # Sort them descending by correlation
    high_corr_pairs.sort(key=lambda x: x[2], reverse=True)

    print(f"\nFound {len(high_corr_pairs)} cross-market pairs with correlation >= {corr_threshold}.\n")

    # ---------------------------------------------------------
    # (C) Run cointegration tests on these high-correlation pairs
    # ---------------------------------------------------------
    coint_results = []
    for (tokA, tokB, corr_val) in high_corr_pairs:
        seriesA = data[tokA].dropna()
        seriesB = data[tokB].dropna()

        # Align on same dates (important if your data has missing values)
        aligned = pd.concat([seriesA, seriesB], axis=1).dropna()
        alignedA = aligned[tokA]
        alignedB = aligned[tokB]

        # Engle–Granger test: returns (test_stat, pvalue, crit_values)
        # H0: "No cointegration"; so pvalue < pval_threshold => cointegrated
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

    # Filter to only cointegrated
    cointegrated_pairs = [res for res in coint_results if res["cointegrated"]]

    print(f"\nCointegration results (p-value < {pval_threshold}):")
    for res in cointegrated_pairs:
        print(
            f"{res['token_1']} & {res['token_2']} | Corr={res['corr']:.3f} | p-value={res['pvalue']:.4f} "
            f"(crit_values={res['crit_values']}) -> COINTEGRATED"
        )

    # Optionally convert results to DataFrame and return/sort/save
    coint_df = pd.DataFrame(coint_results).sort_values("pvalue", ascending=True)

    # Return a dictionary with the single max correlation result and the full cointegration table
    return {
        # "max_correlation_result": {
        #     "max_correlation": max_value,
        #     "token_1": token_1,
        #     "token_2": token_2,
        #     "market_id_1": market_id_1,
        #     "market_id_2": market_id_2,
        # },
        "coint_results": coint_df
    }

if __name__ == "__main__":
    directory = "/Users/ru/Polymarket/time_series"
    
    # Load all market time-series into a single DataFrame
    combined_data = market_data(directory)
    
    if not combined_data.empty:
        print("Data loaded. Head of the combined DataFrame:")
        print(combined_data.head())
        
        # Calculate & visualize cross-market correlations, then check cointegration
        result = filter_and_calculate_correlations_masked(
            combined_data,
            corr_threshold=0.8,
            pval_threshold=0.05
        )
        
        # Save the maximum correlation pair to CSV
        max_corr_df = pd.DataFrame([result["max_correlation_result"]])
        max_corr_df.to_csv("max_correlation_result.csv", index=False)

        # Save the full cointegration test results to CSV
        result["coint_results"].to_csv("cointegration_results.csv", index=False)

        print("Maximum correlation result saved to 'max_correlation_result.csv'.")
        print("Cointegration results saved to 'cointegration_results.csv'.")
    else:
        print("No data was loaded. Ensure the directory contains valid CSV files.")
