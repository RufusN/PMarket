#!/usr/bin/env python3

import sys
import json
import time
import os
import requests
import argparse
from datetime import datetime
from py_clob_client.client import ClobClient
from dotenv import load_dotenv

# Load environment variables for API keys
load_dotenv()
api_key = os.getenv('API_KEY')

# Initialize Polymarket client
host = "https://clob.polymarket.com"
chain_id = 137  # Polygon Mainnet
client = ClobClient(host, key=api_key, chain_id=chain_id)

# Constants
MINIMUM_SAMPLE_SIZE = 20
MIN_TIME_INTERVAL = 1  # Minimum time interval in minutes
MIN_VOLUME = 1000  # Minimum volume for a market to be considered

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

def fetch_order_book(token_id):
    """Fetch order book for a token and calculate total volume."""
    try:
        print(f"Fetching order book for token_id: {token_id}")
        order_book = client.get_order_book(token_id)

        if not hasattr(order_book, 'bids') or not hasattr(order_book, 'asks'):
            print(f"Order book for token_id {token_id} is missing bids or asks.")
            return 0

        total_volume = sum(float(order.size) for order in order_book.bids + order_book.asks)
        return total_volume
    except Exception as e:
        print(f"Error fetching order book for token_id {token_id}: {e}")
        return 0

def fetch_time_series(token_id, start_ts, end_ts, fidelity, attempt=1, max_attempts=3, wait_time=1):
    """Fetch time-series data with retry logic."""
    endpoint = f"{host}/prices-history"
    params = {
        "market": token_id,
        "startTs": start_ts,
        "endTs": end_ts,
        "fidelity": fidelity
    }

    try:
        print(f"Fetching time-series data for token_id: {token_id} (attempt {attempt})...")
        response = requests.get(endpoint, params=params)
        response.raise_for_status()
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
        print(f"Error fetching time series data for token_id {token_id}: {e}")
        return []

def process_markets_and_calculate_volume(markets, start_ts, end_ts, fidelity):
    """Processes markets to calculate trade volumes and retrieve token time-series data."""
    processed_data = []
    for market in markets:
        market_id = market.get("condition_id")
        market_name = market.get("question", "Unknown Market")
        market_tags = market.get("tags", [])
        market_end_date = market.get("end_date_iso", "Unknown End Date")

        print(f"\nProcessing market: {market_name} (ID: {market_id})")
        print(f"Tags: {market_tags}, End Date: {market_end_date}")

        tokens = market.get('tokens', [])
        if not tokens:
            print(f"No tokens found for market: {market_name}. Skipping.")
            continue

        tokens_data = []
        total_volume = 0
        for token in tokens:
            token_id = token.get("token_id")
            if not token_id:
                print(f"Token ID missing for market: {market_name}. Skipping this token.")
                continue

            # Fetch order book volume
            volume = fetch_order_book(token_id)
            total_volume += volume

            # Fetch time-series data for the token
            ts_data = fetch_time_series(token_id, start_ts, end_ts, fidelity)

            # Process time-series data
            # if ts_data:
            #     try:
            #         import pandas as pd

            #         # Convert time series data to a Pandas DataFrame
            #         df = pd.DataFrame(ts_data)
            #         df.rename(columns={"t": "timestamp", "p": "price"}, inplace=True)
            #         df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
            #         df.set_index("timestamp", inplace=True)

            #         # Remove duplicate timestamps by keeping the first occurrence
            #         duplicate_count = df.index.duplicated().sum()
            #         if duplicate_count > 0:
            #             print(f"Token {token_id} has {duplicate_count} duplicate timestamps. Removing duplicates.")
            #             df = df[~df.index.duplicated(keep="first")]

            #         # Check for zero variance
            #         if df["price"].std() == 0:
            #             print(f"Token {token_id} has zero variance - skipping this token.")
            #             continue  # Skip this token

            #         # Convert back to list of dicts for JSON compatibility
            #         ts_data = [{"t": int(ts.timestamp()), "p": price} for ts, price in df["price"].items()]
            #     except Exception as e:
            #         print(f"Error processing time series for token_id {token_id}: {e}")
            #         continue

            tokens_data.append({
                "token_id": token_id,
                "time_series": ts_data
            })

        # Append market data with tokens
        processed_data.append({
            "market_id": market_id,
            "market_name": market_name,
            "tags": market_tags,
            "end_date": market_end_date,
            "total_volume": total_volume,
            "tokens": tokens_data
        })

    print(f"Processed {len(processed_data)} markets.")
    return processed_data

def save_to_json(data, output_file="sorted_markets_with_tokens.json"):
    """Save processed market data to a JSON file."""
    with open(output_file, 'w') as file:
        json.dump(data, file, indent=4)
    print(f"Data saved to {output_file}")

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Fetch, process, and sort Polymarket data by trade volume."
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
    parser.add_argument(
        "--output", type=str, default="sorted_markets_with_tokens.json",
        help="Output JSON file to save sorted markets (default: sorted_markets_with_tokens.json)."
    )
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_arguments()
    time_interval_in_minutes = args.time_interval_in_minutes
    duration_in_weeks = args.duration_in_weeks
    output_file = args.output
    print(f"Time Interval: {time_interval_in_minutes} minute(s)")
    if duration_in_weeks is not None:
        print(f"Duration: Last {duration_in_weeks} week(s)")
    else:
        print("Duration: Entire history (start_time=0)")

    # Calculate start and end times
    end_time = int(time.time())
    start_time = end_time - (int(duration_in_weeks * 7 * 24 * 60 * 60) if duration_in_weeks else 0)

    print(f"Fetching active markets...")
    markets = fetch_markets()
    tradable_markets = [m for m in markets if m.get('enable_order_book', True)]

    if not tradable_markets:
        print("No tradable markets found.")
        sys.exit(0)

    print("Processing markets to calculate volumes and fetch time-series data...")
    processed_markets = process_markets_and_calculate_volume(tradable_markets, start_time, end_time, time_interval_in_minutes)

    print("Sorting markets by total trade volume...")
    sorted_markets = sorted(processed_markets, key=lambda x: x["total_volume"], reverse=True)

    sorted_markets = sorted_markets[:800]  # Limit to a minimum sample size

    print("Saving sorted markets to JSON...")
    save_to_json(sorted_markets, output_file=output_file)
