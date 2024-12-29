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

def fetch_time_series(token_id, start_ts, end_ts, fidelity=60):
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
    print("Fetching markets...")
    markets = fetch_markets()

    # Filter tradable markets
    tradable_markets = [
        m for m in markets if m.get('enable_order_book', True)
    ]
    
    if not tradable_markets:
        print("No tradable markets found.")
        exit()

    # Limit to the first 5 tradable markets
    tradable_markets = tradable_markets[:5]

    # Calculate timestamps for the past 2 days
    end_time = int(time.time())
    start_time = end_time - (2 * 24 * 60 * 60)  # Past 2 days

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
            time_series_data = fetch_time_series(token_id, start_ts=start_time, end_ts=end_time, fidelity=60)

            if time_series_data:
                write_to_csv(time_series_data, market_id, market_name, token_id)
            else:
                print(f"No data available for token {token_id}.")
