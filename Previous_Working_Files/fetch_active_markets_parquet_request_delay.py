#!/usr/bin/env python3

import sys
import json
import time
import requests
import pandas as pd
from datetime import datetime
from py_clob_client.client import ClobClient
from keys import pass_key  # Import the API key
import math
import argparse  # Import for command-line argument parsing

# Constants
MINIMUM_SAMPLE_SIZE = 20
MIN_TIME_INTERVAL = 1  # Minimum time interval in minutes
DEFAULT_MAX_THREADS = 10  # Default number of threads (not used in non-parallel script)
MIN_VOLUME = 1000  # Minimum volume for a market to be considered

# Initialize client
host = "https://clob.polymarket.com"
chain_id = 137  # Polygon Mainnet

client = ClobClient(
    host,
    key=pass_key,
    chain_id=chain_id
)

def fetch_markets():
    """Fetches all markets from the API, handling pagination."""
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
            # if 'data' in response and response['data']:
            #     print(response.values())
            #     break

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

def parse_arguments():
    """
    Parses command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Fetch and process time series data for tokens from Polymarket."
    )
    parser.add_argument(
        "time_interval_in_minutes",
        nargs='?',
        type=int,
        default=MIN_TIME_INTERVAL,
        help=f"Time interval in minutes (default: {MIN_TIME_INTERVAL}). Must be a positive integer."
    )
    parser.add_argument(
        "duration_in_weeks",
        nargs='?',
        type=float,
        default=None,
        help="Duration in weeks to fetch data for (default: entire history). Must be a positive number."
    )
    return parser.parse_args()

if __name__ == "__main__":
    # -----------------------------------------------------------------------
    # USAGE:
    # python X.py [time_interval_in_minutes] [duration_in_weeks]
    #
    # - If time_interval_in_minutes is omitted, use MIN_TIME_INTERVAL.
    # - If duration_in_weeks is omitted, fetch the entire history (start_time=0).
    # -----------------------------------------------------------------------

    # 1) Optional: time_interval_in_minutes
    args = parse_arguments()
    time_interval_in_minutes = args.time_interval_in_minutes  # Default value
    duration_in_weeks = args.duration_in_weeks  # Default value

    if len(sys.argv) >= 2:
        try:
            time_interval_input = int(sys.argv[1])
            if time_interval_input <= 0:
                raise ValueError("time_interval_in_minutes must be a positive integer.")
            time_interval_in_minutes = max(time_interval_input, MIN_TIME_INTERVAL)
            if time_interval_input < MIN_TIME_INTERVAL:
                print(f"Provided time_interval_in_minutes ({time_interval_input}) is less than the minimum "
                      f"({MIN_TIME_INTERVAL}). Using {MIN_TIME_INTERVAL} instead.")
        except ValueError as e:
            print(f"Error with time_interval_in_minutes: {e}")
            sys.exit(1)

    # 2) Optional: duration_in_weeks
    #    If not provided, use "max" (start_time=0).
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
    print(f"Time Interval: {time_interval_in_minutes} minute(s)")
    if duration_in_weeks is not None:
        print(f"Duration: Last {duration_in_weeks} week(s)")
    else:
        print("Duration: Entire history (start_time=0)")
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
    tradable_markets = tradable_markets[:100]  # This line doesn't alter the list; adjust as needed

    # Prepare a list to hold all rows
    all_data_rows = []

    for market in tradable_markets:
        market_id = market.get('condition_id')
        market_name = market.get('question', 'unknown_market')
        market_tags = market.get('tags', [])
        market_end_date = market.get('end_date_iso', 'unknown_date')
        market_vol = market.get('volume_num_min', 'unknown_volume')

        # extracted_market = {
        #             "market_id": market.get("condition_id"),
        #             "market_name": market.get("question"),
        #             "description": market.get("description"),
        #             "tags": market.get("tags"),
        #             "end_date": market.get("end_date_iso"),
        #             "volume": sum(token.get("price", 0) for token in market.get("tokens", [])),  # Example aggregation
        #         }
        print(f"\nProcessing market: {market_name} (ID: {market_id})")
        print(f"\nMarket tags: {market_tags} (End date: {market_end_date})")
        print(market_vol)
        
        tokens = market.get('tokens', [])
        if not tokens:
            print(f"No tokens found for market: {market_name}")
            continue

        for token in tokens:
            token_id = token.get("token_id")
            if not token_id:
                print(f"No token ID for token in market: {market_name}")
                continue

            # Fetch time series data
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
                continue  # Skip this token

            # Calculate actual number of data points
            actual_num_points = len(time_series_data)

            # Validate the length of the fetched data
            if actual_num_points < MINIMUM_SAMPLE_SIZE:
                # Optionally, log this event
                print(f"Token {token_id} has fewer data points ({actual_num_points}) than minimum required ({MINIMUM_SAMPLE_SIZE}). Skipping this token.")
                continue  # Skip this token

            # Accumulate timestamps and prices
            timestamps = []
            prices = []
            for point in time_series_data:
                timestamps.append(point["t"])
                prices.append(point["p"])

            # Create a pandas Series for the price, indexed by timestamp
            try:
                # Convert Unix timestamps to pandas datetime
                series = pd.Series(data=prices, index=pd.to_datetime(timestamps, unit='s'))
            except Exception as e:
                print(f"Error converting timestamps to datetime for token {token_id}: {e} - skipping.")
                continue  # Skip this token

            # # Check if there are at least two timestamps to compare
            # if len(series) >= 2:
            #     # Compare the last timestamp with the penultimate timestamp
            #     last_timestamp = series.index[-1]
            #     penultimate_timestamp = series.index[-2]
            #     if last_timestamp == penultimate_timestamp:
            #         # Remove the last timestamp
            #         series = series.iloc[:-1]
            #         print('Duplicate found and removed (-1 timestep)')
                    # print(f"Token {token_id}: Removed the last timestamp as it duplicates the penultimate timestamp.")

            # Remove duplicate timestamps by keeping the first occurrence
            duplicate_count = series.index.duplicated().sum()
            if duplicate_count > 0:
                print(f"Token {token_id} has {duplicate_count} duplicate timestamps. Removing duplicates.")
                series = series[~series.index.duplicated(keep='first')]

            # Check for zero variance
            if series.std() == 0:
                print(f"Token {token_id} has zero variance - skipping this token.")
                continue  # Skip this token

            # If all checks pass, add a single entry with arrays for timestamps and prices
            all_data_rows.append({
                "market_id": market_id,
                "market_name": market_name,
                "token_id": token_id,
                "timestamps": timestamps,  # List of integers
                "prices": prices  # List of floats
            })

    # Once done, convert to a DataFrame and write to Parquet
    if all_data_rows:
        df = pd.DataFrame(all_data_rows)
        parquet_filename = "time_series_data.parquet"
        try:
            df.to_parquet(parquet_filename, index=False, engine='pyarrow')
            print(f"\nAll time series data written to {parquet_filename}.")
        except Exception as e:
            print(f"Error saving Parquet file: {e}")
    else:
        print("\nNo data was collected - parquet file not created.")
