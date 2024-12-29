import csv
import json
import time
import requests
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

def fetch_time_series(token_id, interval="1d"):
    endpoint = f"{host}/prices-history"
    params = {"market": token_id, "interval": interval}
    try:
        response = requests.get(endpoint, params=params)
        print(f"Request URL: {response.url}")
        response.raise_for_status()
        data = response.json()
        print(f"API Response: {json.dumps(data, indent=2)}")
        return data.get('history', [])
    except requests.exceptions.RequestException as e:
        print(f"Error fetching time series data: {e}")
        return []

if __name__ == "__main__":
    print("Fetching markets...")
    markets = fetch_markets()

    # Filter markets with `enable_order_book == True`
    tradable_markets = [
        m for m in markets if m.get('enable_order_book', True)
    ]
    
    if not tradable_markets:
        print("No tradable markets found.")
        exit()

    for market in tradable_markets:
        print(f"Processing market: {market.get('question')}")

        # Check if the market has tokens
        tokens = market.get('tokens', [])
        if not tokens:
            print(f"No tokens found for market: {market.get('question')}")
            continue

        for token in tokens:
            token_id = token.get("token_id")
            if not token_id:
                print(f"No token ID found for token in market: {market.get('question')}")
                continue

            # Fetch time series data for the token
            print(f"Fetching time series data for token ID: {token_id}")
            time_series_data = fetch_time_series(token_id)

            if not time_series_data:
                print(f"No time series data found for token ID: {token_id}")
                continue

            # Output to CSV
            csv_file = f"timeseries_token_{token_id}.csv"
            try:
                with open(csv_file, 'w', newline='') as csvfile:
                    fieldnames = ['timestamp', 'price']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()

                    for point in time_series_data:
                        writer.writerow({'timestamp': point['t'], 'price': point['p']})
                
                print(f"Time series data written to {csv_file}.")
            except IOError as e:
                print(f"Error writing to CSV: {e}")
