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
import argparse  # For command-line argument parsing

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
            # print(f"Fetching markets with next_cursor: {next_cursor}")
            if next_cursor is None:
                response = client.get_markets()
            else:
                response = client.get_markets(next_cursor=next_cursor)

            if 'data' not in response or not response['data']:
                print("No data found in response.")
                break

            markets_list.extend(response['data'])
            next_cursor = response.get("next_cursor", None)

            if not next_cursor or next_cursor == "LTE=":
                break
        except Exception as e:
            print(f"Error fetching markets: {e}")
            break
    print(f"Fetched {len(markets_list)} markets.")
    return markets_list

def fetch_time_series(token_id, start_ts, end_ts, fidelity, attempt=1, max_attempts=3, wait_time=1):
    """
    Fetch time-series data with a simple backoff strategy:
    """
    endpoint = f"{host}/prices-history"
    params = {
        "market": token_id,
        "startTs": start_ts,
        "endTs": end_ts,
        "fidelity": fidelity
    }

    try:
        # print(f"Fetching time-series data for token_id: {token_id} (attempt {attempt})...")
        response = requests.get(endpoint, params=params)
        response.raise_for_status()  # Raise an HTTPError for status codes >= 400
        data = response.json()
        return data.get('history', [])

    except requests.exceptions.HTTPError as http_err:
        if response.status_code == 429 and attempt <= max_attempts:
            print(f"429 Too Many Requests for token_id: {token_id}. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
            return fetch_time_series(token_id, start_ts, end_ts, fidelity,
                                     attempt=attempt+1, max_attempts=max_attempts, wait_time=wait_time*2)
        else:
            print(f"HTTP Error: {http_err} for token_id: {token_id}")
            return []
    except requests.exceptions.RequestException as e:
        print(f"Error fetching time series data: {e}")
        return []

def calculate_expected_points(start_ts, end_ts, fidelity_minutes):
    """Calculate the expected number of data points."""
    total_seconds = end_ts - start_ts
    fidelity_seconds = fidelity_minutes * 60
    expected_points = math.floor(total_seconds / fidelity_seconds) + 1  # +1 to include both start and end
    return expected_points

def save_to_json(data, output_file="markets_with_time_series.json"):
    """Save processed market data to a JSON file."""
    with open(output_file, 'w') as file:
        json.dump(data, file, indent=4)
    print(f"Data saved to {output_file}")

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Fetch and process time series data for tokens from Polymarket."
    )
    parser.add_argument(
        "time_interval_in_minutes",
        nargs='?',
        type=int,
        default=MIN_TIME_INTERVAL,
        help=f"Time interval in minutes (default: {MIN_TIME_INTERVAL})."
    )
    parser.add_argument(
        "duration_in_weeks",
        nargs='?',
        type=float,
        default=None,
        help="Duration in weeks to fetch data for (default: entire history)."
    )
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_arguments()
    time_interval_in_minutes = args.time_interval_in_minutes
    duration_in_weeks = args.duration_in_weeks

    # Calculate start and end times
    end_time = int(time.time())
    start_time = end_time - (int(duration_in_weeks * 7 * 24 * 60 * 60) if duration_in_weeks else 0)

    print(f"Fetching active markets...")
    markets = fetch_markets()
    tradable_markets = [m for m in markets if m.get('enable_order_book', True)]

    if not tradable_markets:
        print("No tradable markets found.")
        sys.exit(0)

    # Prepare data structure for JSON
    all_data_rows = []

    for market in tradable_markets:
        market_id = market.get('condition_id')
        market_name = market.get('question', 'Unknown Market')
        market_tags = market.get('tags', [])
        market_end_date = market.get('end_date_iso', 'Unknown End Date')

        print(f"\nProcessing market: {market_name} (ID: {market_id})")
        print(f"Tags: {market_tags}, End Date: {market_end_date}")

        tokens = market.get('tokens', [])
        if not tokens:
            print(f"No tokens found for market: {market_name}. Skipping.")
            continue

        total_volume = 0
        time_series_data = []

        for token in tokens:
            token_id = token.get("token_id")
            if not token_id:
                print(f"Token ID missing for market: {market_name}. Skipping this token.")
                continue

            # Fetch time-series data
            ts_data = fetch_time_series(token_id, start_time, end_time, time_interval_in_minutes)
            if ts_data:
                time_series_data.extend(ts_data)

        # Collect market data
        all_data_rows.append({
            "market_id": market_id,
            "market_name": market_name,
            "tags": market_tags,
            "end_date": market_end_date,
            "time_series": time_series_data
        })

    # Save data to JSON
    save_to_json(all_data_rows)
