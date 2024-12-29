import sys
import json
import time
import requests
import pandas as pd
from datetime import datetime
from py_clob_client.client import ClobClient
from keys import pass_key  # Import the API key
import math

MINIMUM_SAMPLE_SIZE = 20

# Initialize client
host = "https://clob.polymarket.com"
chain_id = 137  # Polygon Mainnet

client = ClobClient(
    host,
    key=pass_key,
    chain_id=chain_id
)

def fetch_markets():
    markets_list = []
    next_cursor = None
    while True:
        try:
            if next_cursor is None:
                response = client.get_markets()
            else:
                response = client.get_markets(next_cursor=next_cursor)
            
            next_cursor = response.get("next_cursor", None)
            if 'data' not in response or not response['data']:
                print("No data found in response.")
                break

            markets_list.extend(response['data'])

            if not next_cursor or next_cursor == "LTE=":
                break
        except Exception as e:
            print(f"Error fetching markets: {e}")
            break
    return markets_list

def fetch_time_series(token_id, start_ts, end_ts, fidelity, attempt=1, max_attempts=3, wait_time=1):
    """
    Fetch time-series data with a simple backoff strategy:
      - If we hit a 429 (Too Many Requests), wait 'wait_time' seconds and then retry.
      - Each retry doubles the wait_time. 
      - Stop after 'max_attempts' attempts.
    """
    endpoint = f"{host}/prices-history"
    params = {
        "market": token_id,
        "startTs": start_ts,
        "endTs": end_ts,
        "fidelity": fidelity
    }

    try:
        response = requests.get(endpoint, params=params)
        # Raise an HTTPError for status codes >= 400
        response.raise_for_status()
        data = response.json()
        return data.get('history', [])

    except requests.exceptions.HTTPError as http_err:
        # Check if it's a 429 Too Many Requests
        if response is not None and response.status_code == 429:
            if attempt <= max_attempts:
                print(f"Received 429 Too Many Requests. Retrying attempt {attempt} of {max_attempts} "
                      f"after sleeping {wait_time} seconds...")
                time.sleep(wait_time)
                # Exponential backoff: wait_time * 2
                return fetch_time_series(token_id, start_ts, end_ts, fidelity,
                                         attempt=attempt+1,
                                         max_attempts=max_attempts,
                                         wait_time=wait_time * 2)
            else:
                print(f"Max retries reached for token_id: {token_id}")
                return []
        else:
            # Some other HTTP error—log and return empty or raise
            print(f"HTTP Error: {http_err} for token_id: {token_id}")
            return []

    except requests.exceptions.RequestException as e:
        print(f"Error fetching time series data: {e}")
        return []

def calculate_expected_points(start_ts, end_ts, fidelity_minutes):
    """
    Calculate the expected number of data points based on the time range and fidelity.
    """
    total_seconds = end_ts - start_ts
    fidelity_seconds = fidelity_minutes * 60
    expected_points = math.floor(total_seconds / fidelity_seconds) + 1  # +1 to include both start and end
    return expected_points

if __name__ == "__main__":
    # -----------------------------------------------------------------------
    # USAGE:
    # python X.py <time_interval_in_minutes> [duration_in_weeks]
    #
    # If duration_in_weeks is omitted, we fetch the entire history (start_time=0).
    # -----------------------------------------------------------------------

    if len(sys.argv) < 2:
        print("Usage: python X.py <time_interval_in_minutes> [duration_in_weeks]")
        sys.exit(1)

    # 1) Required: time_interval_in_minutes
    try:
        time_interval_in_minutes = int(sys.argv[1])
        if time_interval_in_minutes <= 0:
            raise ValueError("time_interval_in_minutes must be a positive integer.")
    except ValueError as e:
        print(f"Error with time_interval_in_minutes: {e}")
        sys.exit(1)

    # 2) Optional: duration_in_weeks
    #    If not provided, we fetch from start_time=0 (i.e. earliest data).
    duration_in_weeks = None
    if len(sys.argv) > 2:
        try:
            duration_in_weeks = float(sys.argv[2])
            if duration_in_weeks <= 0:
                raise ValueError("duration_in_weeks must be positive if provided.")
        except ValueError as e:
            print(f"Error with duration_in_weeks: {e}")
            sys.exit(1)

    # Calculate the time range
    end_time = int(time.time())
    if duration_in_weeks is not None:
        # Convert weeks to seconds
        duration_in_seconds = int(duration_in_weeks * 7 * 24 * 60 * 60)
        start_time = end_time - duration_in_seconds
    else:
        # If duration not provided, use "max" (start_time=0)
        start_time = 0

    # Calculate expected number of data points
    expected_num_points = calculate_expected_points(start_time, end_time, time_interval_in_minutes)
    print(f"Expected number of data points per token: {expected_num_points}")

    # Fetch markets
    print("\nFetching markets...")
    markets = fetch_markets()

    # Filter tradable markets
    tradable_markets = [
        m for m in markets if m.get('enable_order_book', True)
    ]
    
    if not tradable_markets:
        print("No tradable markets found.")
        sys.exit(0)

    # Example: use all tradable markets (remove or reduce as needed)
    tradable_markets = tradable_markets[:200]

    # Prepare a list to hold all rows
    all_data_rows = []

    for market in tradable_markets:
        market_id = market.get('condition_id')
        market_name = market.get('question', 'unknown_market')
        print(f"\nProcessing market: {market_name} (ID: {market_id})")
        
        tokens = market.get('tokens', [])
        if not tokens:
            print(f"No tokens found for market: {market_name}")
            continue

        for token in tokens:
            token_id = token.get("token_id")
            if not token_id:
                print(f"No token ID for token in market: {market_name}")
                continue

            # print(f"Fetching time series data for token {token_id}...")
            time_series_data = fetch_time_series(
                token_id=token_id,
                start_ts=start_time,
                end_ts=end_time,
                fidelity=time_interval_in_minutes
            )

            # If there's no data returned, the market might not have existed that long,
            # or we reached max retries for 429. 
            if not time_series_data:
                if duration_in_weeks is not None:
                    print(f"No data for token {token_id} within the last {duration_in_weeks} weeks.")
                else:
                    print(f"No data for token {token_id}. (Older or inactive market, or request failed.)")
                continue

            # Calculate actual number of data points
            actual_num_points = len(time_series_data)
            # print(f"Fetched {actual_num_points} data points for token {token_id}.")

            # Validate the length of the fetched data
            # if actual_num_points < expected_num_points:
            if actual_num_points < MINIMUM_SAMPLE_SIZE:
                # print(f"Error: Token {token_id} has fewer data points ({actual_num_points}) than expected ({expected_num_points}). Skipping this token.")
                continue  # Skip this token

            # If desired, you can allow a small tolerance, e.g., allow missing up to 5%
            # tolerance = 0.95
            # if actual_num_points < expected_num_points * tolerance:
            #     print(f"Error: Token {token_id} has fewer data points ({actual_num_points}) than expected ({expected_num_points}). Skipping this token.")
            #     continue

            timestamps = []
            prices = []
            # Accumulate rows in our list
            # time_series_data_sorted = sorted(time_series_data, key=lambda x: x["t"])
            for point in time_series_data:
                timestamps.append(point["t"])
                prices.append(point["p"])
            all_data_rows.append({
                "market_id": market_id,
                "market_name": market_name,
                "token_id": token_id,
                "timestamp": timestamps,
                "price": prices
            })

    # Once done, convert to a DataFrame and write to Parquet
    if all_data_rows:
        df = pd.DataFrame(all_data_rows)
        parquet_filename = "time_series_data.parquet"
        try:
            df.to_parquet(parquet_filename, index=False)
            print(f"\nAll time series data written to {parquet_filename}.")
        except Exception as e:
            print(f"Error saving Parquet file: {e}")
    else:
        print("\nNo data was collected - parquet file not created.")
