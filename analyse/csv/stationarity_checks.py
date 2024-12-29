import pandas as pd
from statsmodels.tsa.stattools import adfuller
# from statsmodels.tsa.stattools import kpss  # Uncomment if you want to use KPSS

def adf_test(series, significance_level=0.05):
    """
    Perform the Augmented Dickey-Fuller test to check if 'series' is stationary.

    :param series: A pandas Series or list-like of time-series data.
    :param significance_level: Significance level for deciding stationarity.
    :return: A tuple (is_stationary, p_value, test_stat, critical_values)
             where is_stationary is True/False based on the p-value.
    """
    # Drop missing values to avoid errors
    series = series.dropna()

    result = adfuller(series, autolag='AIC')
    test_stat = result[0]
    p_value = result[1]
    crit_values = result[4]

    is_stationary = p_value < significance_level

    print("===== ADF Test =====")
    print(f"ADF Statistic: {test_stat:.4f}")
    print(f"p-value: {p_value:.4f}")
    print(f"Critical Values: {crit_values}")
    print(f"Stationary at {significance_level} level? {'YES' if is_stationary else 'NO'}")
    print("====================\n")

    return is_stationary, p_value, test_stat, crit_values

# def kpss_test(series, significance_level=0.05, regression='c'):
#     """
#     Perform the KPSS test to check if 'series' is stationary.
#
#     :param series: A pandas Series or list-like of time-series data.
#     :param significance_level: Significance level for deciding stationarity.
#     :param regression: Either 'c' (constant) or 'ct' (constant + trend).
#     :return: A tuple (is_stationary, p_value, test_stat, critical_values)
#              where is_stationary is True/False based on the p-value.
#     """
#     series = series.dropna()
#     statistic, p_value, n_lags, crit_values = kpss(series, regression=regression)
#
#     # For KPSS, Null Hypothesis = "Data is stationary"
#     is_stationary = p_value > significance_level
#
#     print("===== KPSS Test =====")
#     print(f"KPSS Statistic: {statistic:.4f}")
#     print(f"p-value: {p_value:.4f}")
#     print(f"Critical Values: {crit_values}")
#     print(f"Stationary at {significance_level} level? {'YES' if is_stationary else 'NO'}")
#     print("====================\n")
#
#     return is_stationary, p_value, statistic, crit_values

def check_stationarity(series, alpha=0.05):
    """
    Check the stationarity of a time series using ADF (and optionally KPSS).
    Returns a dictionary with the results.

    :param series: A pandas Series or list-like of time-series data.
    :param alpha: Significance level for stationarity decisions.
    :return: A dict with ADF results, and optionally KPSS results if you uncomment the code.
    """
    results = {}

    # --- Run ADF ---
    adf_stationary, adf_p, adf_stat, adf_crit = adf_test(series, alpha)
    results["ADF_Stationary"] = adf_stationary
    results["ADF_p_value"] = adf_p
    results["ADF_Statistic"] = adf_stat
    results["ADF_CriticalValues"] = adf_crit

    # --- (Optional) Run KPSS ---
    # kpss_stationary, kpss_p, kpss_stat, kpss_crit = kpss_test(series, alpha)
    # results["KPSS_Stationary"] = kpss_stationary
    # results["KPSS_p_value"] = kpss_p
    # results["KPSS_Statistic"] = kpss_stat
    # results["KPSS_CriticalValues"] = kpss_crit

    return results

if __name__ == "__main__":
    # Example usage:
    # Suppose you have some DataFrame 'df' with a time-series column 'price'.
    # We'll simulate some data here for demonstration:

    import numpy as np

    # Create a random walk (often ~ I(1)) plus some trend
    np.random.seed(42)
    steps = 100
    random_walk = np.cumsum(np.random.randn(steps)) + 100
    df = pd.DataFrame({
        "price": random_walk
    })

    # Run the stationarity checks
    analysis_results = check_stationarity(df["price"], alpha=0.05)

    # If you uncomment the KPSS part in the code, you'll get both results:
    # - ADF: Null hypothesis = "Series is non-stationary (has a unit root)"
    # - KPSS: Null hypothesis = "Series is stationary"
    # They complement each other.
    
    print("Final Summary:")
    print(analysis_results)
