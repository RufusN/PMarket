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
import concurrent.futures  # Import for multithreading
import argparse  # Import for command-line argument parsing

# Constants
MINIMUM_SAMPLE_SIZE = 20
MIN_TIME_INTERVAL = 1  # Minimum time interval in minutes
DEFAULT_MAX_THREADS = 10  # Default number of threads

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

            markets_list.extend(response['data'])

            if not next_cursor or next_cursor == "LTE=":
                break
        except Exception as e:
            print(f"Error fetching markets: {e}")
            break
    return markets_list

def fetch_time_series(session, token_id, start_ts, end_ts, fidelity, attempt=1, max_attempts=3, wait_time=1):
    """
    Fetches time-series data for a given token with an exponential backoff strategy.
    
    Args:
        session (requests.Session): The session object for connection pooling.
        token_id (str): The ID of the token to fetch data for.
        start_ts (int): Start timestamp (Unix epoch).
        end_ts (int): End timestamp (Unix epoch).
        fidelity (int): Time interval in minutes.
        attempt (int, optional): Current retry attempt. Defaults to 1.
        max_attempts (int, optional): Maximum number of retry attempts. Defaults to 3.
        wait_time (int, optional): Initial wait time in seconds before retrying. Defaults to 1.
    
    Returns:
        list: A list of time-series data points or an empty list on failure.
    """
    endpoint = f"{host}/prices-history"
    params = {
        "market": token_id,
        "startTs": start_ts,
        "endTs": end_ts,
        "fidelity": fidelity
    }

    try:
        response = session.get(endpoint, params=params)
        # Raise an HTTPError for status codes >= 400
        response.raise_for_status()
        data = response.json()
        return data.get('history', [])

    except requests.exceptions.HTTPError as http_err:
        # Check if it's a 429 Too Many Requests
        if response is not None and response.status_code == 429:
            if attempt <= max_attempts:
                print(f"Received 429 Too Many Requests for token {token_id}. Retrying attempt {attempt} of {max_attempts} after sleeping {wait_time} seconds...")
                time.sleep(wait_time)
                # Exponential backoff: wait_time * 2
                return fetch_time_series(session, token_id, start_ts, end_ts, fidelity,
                                         attempt=attempt+1,
                                         max_attempts=max_attempts,
                                         wait_time=wait_time * 2)
            else:
                print(f"Max retries reached for token_id: {token_id}. Skipping this token.")
                return []
        else:
            # Some other HTTP error—log and return empty
            print(f"HTTP Error: {http_err} for token_id: {token_id}. Skipping this token.")
            return []

    except requests.exceptions.RequestException as e:
        print(f"Error fetching time series data for token {token_id}: {e}. Skipping this token.")
        return []

def calculate_expected_points(start_ts, end_ts, fidelity_minutes):
    """
    Calculates the expected number of data points based on the time range and fidelity.
    
    Args:
        start_ts (int): Start timestamp (Unix epoch).
        end_ts (int): End timestamp (Unix epoch).
        fidelity_minutes (int): Time interval in minutes.
    
    Returns:
        int: Expected number of data points.
    """
    total_seconds = end_ts - start_ts
    fidelity_seconds = fidelity_minutes * 60
    expected_points = math.floor(total_seconds / fidelity_seconds) + 1  # +1 to include both start and end
    return expected_points

def process_token(session, token, market_id, market_name, start_time, end_time, fidelity, duration_in_weeks):
    """
    Processes a single token: fetches time-series data, cleans it, and returns the structured data.
    
    Args:
        session (requests.Session): The session object for connection pooling.
        token (dict): Token data dictionary.
        market_id (str): ID of the market.
        market_name (str): Name of the market.
        start_time (int): Start timestamp (Unix epoch).
        end_time (int): End timestamp (Unix epoch).
        fidelity (int): Time interval in minutes.
        duration_in_weeks (float or None): Duration in weeks or None for full history.
    
    Returns:
        dict: A dictionary representing the token's time-series data with arrays for timestamps and prices, or an empty dictionary if the token is to be skipped.
    """
    token_id = token.get("token_id")
    if not token_id:
        print(f"No token ID for token in market: {market_name}")
        return {}  # Return an empty dictionary instead of None

    # Fetch time series data
    time_series_data = fetch_time_series(
        session=session,
        token_id=token_id,
        start_ts=start_time,
        end_ts=end_time,
        fidelity=fidelity
    )

    # If there's no data returned, skip this token
    if not time_series_data:
        if duration_in_weeks is not None:
            print(f"No data for token {token_id} within the last {duration_in_weeks} weeks.")
        else:
            print(f"No data for token {token_id}. (Older or inactive market, or request failed.)")
        return {}  # Return an empty dictionary instead of None

    # Calculate actual number of data points
    actual_num_points = len(time_series_data)

    # Validate the length of the fetched data
    if actual_num_points < MINIMUM_SAMPLE_SIZE:
        print(f"Token {token_id} has fewer data points ({actual_num_points}) than minimum required ({MINIMUM_SAMPLE_SIZE}). Skipping this token.")
        return {}  # Return an empty dictionary instead of None

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
        return {}  # Return an empty dictionary instead of None

    # Check if there are at least two timestamps to compare
    if len(series) >= 2:
        # Compare the last timestamp with the penultimate timestamp
        last_timestamp = series.index[-1]
        penultimate_timestamp = series.index[-2]
        if last_timestamp == penultimate_timestamp:
            # Remove the last timestamp
            series = series.iloc[:-1]
            print(f"Token {token_id}: Removed the last timestamp as it duplicates the penultimate timestamp.")

    # Remove duplicate timestamps by keeping the first occurrence
    duplicate_count = series.index.duplicated().sum()
    if duplicate_count > 0:
        print(f"Token {token_id} has {duplicate_count} duplicate timestamps. Removing duplicates.")
        series = series[~series.index.duplicated(keep='first')]

    # Check for zero variance
    if series.std() == 0:
        print(f"Token {token_id} has zero variance - skipping this token.")
        return {}  # Skip this token

    # After cleaning, ensure that timestamps and prices lists are updated
    cleaned_timestamps = series.index.astype(int) // 10**9  # Convert to Unix timestamps
    cleaned_prices = series.values.tolist()

    # Return a single dictionary with arrays for timestamps and prices
    return {
        "market_id": market_id,
        "market_name": market_name,
        "token_id": token_id,
        "timestamps": cleaned_timestamps.tolist(),  # List of integers
        "prices": cleaned_prices  # List of floats
    }

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
    parser.add_argument(
        "--threads",
        type=int,
        default=DEFAULT_MAX_THREADS,
        help=f"Number of concurrent threads (default: {DEFAULT_MAX_THREADS}). Must be a positive integer."
    )
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_arguments()

    # Assign parsed arguments to variables
    time_interval_in_minutes = args.time_interval_in_minutes
    duration_in_weeks = args.duration_in_weeks
    max_threads = args.threads

    # Validate time_interval_in_minutes
    if time_interval_in_minutes < MIN_TIME_INTERVAL:
        print(f"Provided time_interval_in_minutes ({time_interval_in_minutes}) is less than the minimum ({MIN_TIME_INTERVAL}). Using {MIN_TIME_INTERVAL} instead.")
        time_interval_in_minutes = MIN_TIME_INTERVAL

    # Validate duration_in_weeks
    if duration_in_weeks is not None and duration_in_weeks <= 0:
        print("duration_in_weeks must be a positive number. Exiting.")
        sys.exit(1)

    # Validate max_threads
    if max_threads <= 0:
        print("Number of threads must be a positive integer. Exiting.")
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
    print(f"Number of concurrent threads: {max_threads}")

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
    tradable_markets = tradable_markets[:]  # This line doesn't alter the list; adjust as needed

    # Prepare a list to hold all rows
    all_data_rows = []

    # Initialize ThreadPoolExecutor with user-specified number of threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
        # Create a session for connection pooling
        with requests.Session() as session:
            # Create a dictionary to map futures to token IDs for better error handling
            future_to_token = {}
            
            for market in tradable_markets:
                market_id = market.get('condition_id')
                market_name = market.get('question', 'unknown_market')
                print(f"\nProcessing market: {market_name} (ID: {market_id})")
                
                tokens = market.get('tokens', [])
                if not tokens:
                    print(f"No tokens found for market: {market_name}")
                    continue

                for token in tokens:
                    # Submit the token processing to the thread pool
                    future = executor.submit(
                        process_token,
                        session=session,
                        token=token,
                        market_id=market_id,
                        market_name=market_name,
                        start_time=start_time,
                        end_time=end_time,
                        fidelity=time_interval_in_minutes,
                        duration_in_weeks=duration_in_weeks
                    )
                    token_id = token.get("token_id", "Unknown Token ID")
                    future_to_token[future] = token_id

            # As each thread completes, collect the result
            for future in concurrent.futures.as_completed(future_to_token):
                token_id = future_to_token[future]
                try:
                    result = future.result()
                    if result:
                        all_data_rows.append(result)  # Append the dictionary with arrays
                except Exception as exc:
                    print(f"Token {token_id} generated an exception: {exc}")

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
