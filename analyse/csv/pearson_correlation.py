import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from load_time_series import market_data

def filter_and_calculate_correlations_masked(data):
    """
    1) Compute the correlation matrix for all columns/tokens.
    2) Build a mask that is True where tokens share the same market (i.e., same prefix).
    3) Plot a heatmap, using the mask to hide same-market cells.
    4) Return the correlation matrix (with same-market pairs effectively hidden in the plot).
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

    # 4) Plot the heatmap using seaborn, masking the same-market cells
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        corr_matrix,
        mask=mask,         # hide same-market correlations
        cmap="coolwarm",
        center=0,
        annot=False,       # set True if you want numeric values in cells
        fmt=".2f",
        cbar=True,
        square=False,      # set True if you'd prefer square cells
        linewidths=0.5,
        linecolor='gray'
    )
    plt.title("Cross-Market Token Correlations (Masked Same-Market Pairs)")
    # Rotate labels to prevent overlap
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

    # Find the maximum correlation excluding same-market pairs
    masked_corr = corr_matrix.where(~mask)  # Apply mask
    max_corr = masked_corr.stack().idxmax()  # Get index of maximum value
    max_value = masked_corr.loc[max_corr[0], max_corr[1]]  # Get max correlation value

    # Extract the market IDs and filenames
    market_id_1 = col_to_market[max_corr[0]]
    market_id_2 = col_to_market[max_corr[1]]
    token_1 = max_corr[0]
    token_2 = max_corr[1]

    # Print the result
    print(f"Maximum correlation: {max_value}")
    print(f"Between token: {token_1} (Market ID: {market_id_1})")
    print(f"And token: {token_2} (Market ID: {market_id_2})")

    # Optionally return details as a dictionary
    return {
        "max_correlation": max_value,
        "token_1": token_1,
        "token_2": token_2,
        "market_id_1": market_id_1,
        "market_id_2": market_id_2,
    }

if __name__ == "__main__":
    directory = "/Users/ru/Polymarket/time_series"
    
    # Load all market time-series into a single DataFrame
    combined_data = market_data(directory)
    
    if not combined_data.empty:
        print("Data loaded. Head of the combined DataFrame:")
        print(combined_data.head())
        
        # Calculate & visualize cross-market correlations with masking
        result = filter_and_calculate_correlations_masked(combined_data)
        
        # Save the result to a CSV file
        result_df = pd.DataFrame([result])
        result_df.to_csv("max_correlation_result.csv", index=False)
        print("Maximum correlation result saved to 'max_correlation_result.csv'.")
    else:
        print("No data was loaded. Ensure the directory contains valid CSV files.")
