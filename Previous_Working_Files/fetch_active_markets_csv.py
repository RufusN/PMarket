import sys
import csv
import json
import time
import requests
from datetime import datetime, timedelta
from py_clob_client.client import ClobClient
from keys import pass_key  # Import the API key

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

def fetch_time_series(token_id, start_ts, end_ts, fidelity):
    endpoint = f"{host}/prices-history"
    params = {
        "market": token_id,
        "startTs": start_ts,
        "endTs": end_ts,
        "fidelity": fidelity
    }

    try:
        response = requests.get(endpoint, params=params)
        response.raise_for_status()
        data = response.json()
        return data.get('history', [])
    except requests.exceptions.RequestException as e:
        print(f"Error fetching time series data: {e}")
        return []

def write_to_csv(data, market_id, market_name, token_id):
    csv_file = f"{market_id}_{market_name.replace(' ', '_')}_token_{token_id}.csv"
    try:
        with open(csv_file, 'w', newline='') as csvfile:
            fieldnames = ['timestamp', 'price']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for point in data:
                writer.writerow({'timestamp': point['t'], 'price': point['p']})
        print(f"Data for token {token_id} written to {csv_file}")
    except IOError as e:
        print(f"Error writing to CSV for token {token_id}: {e}")

if __name__ == "__main__":
    # -----------------------------------------------------------------------
    # USAGE:
    # python X.py <time_interval_in_minutes> [duration_in_weeks]
    #
    # If duration_in_weeks is omitted, we fetch the entire history (start_time=0).
    # -----------------------------------------------------------------------

    if len(sys.argv) < 2:
        print("Usage: python fetch_active_markets_TS.py <time_interval_in_minutes> [duration_in_weeks]")
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

    end_time = int(time.time())
    if duration_in_weeks is not None:
        # Convert weeks to seconds
        duration_in_seconds = int(duration_in_weeks * 7 * 24 * 60 * 60)
        start_time = end_time - duration_in_seconds
    else:
        # If duration not provided, use "max" (i.e., start_time=0 for earliest possible data)
        start_time = 0

    # Fetch markets
    print("Fetching markets...")
    markets = fetch_markets()

    # Filter tradable markets
    tradable_markets = [
        m for m in markets if m.get('enable_order_book', True)
    ]
    
    if not tradable_markets:
        print("No tradable markets found.")
        sys.exit(0)

    # Limit to the first X tradable markets (can remove or adjust as needed) -  set to all
    tradable_markets = tradable_markets[:]

    for market in tradable_markets:
        market_id = market.get('condition_id')
        market_name = market.get('question', 'unknown_market')
        print(f"Processing market: {market_name} (ID: {market_id})")
        tokens = market.get('tokens', [])
        if not tokens:
            print(f"No tokens found for market: {market_name}")
            continue

        for token in tokens:
            token_id = token.get("token_id")
            if not token_id:
                print(f"No token ID for token in market: {market_name}")
                continue

            print(f"Fetching time series data for token {token_id}")
            time_series_data = fetch_time_series(
                token_id=token_id,
                start_ts=start_time,
                end_ts=end_time,
                fidelity=time_interval_in_minutes
            )

            # If there's no data returned, the market might not have existed for that duration
            if not time_series_data:
                if duration_in_weeks is not None:
                    print(f"No data for token {token_id} within the last {duration_in_weeks} weeks.")
                else:
                    print(f"No data for token {token_id}. (Could be an older or inactive market.)")
                continue

            write_to_csv(time_series_data, market_id, market_name, token_id)
