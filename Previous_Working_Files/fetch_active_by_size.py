import json
import os
from py_clob_client.client import ClobClient
from dotenv import load_dotenv

# Load environment variables for API keys
load_dotenv()
api_key = os.getenv('API_KEY')

# Initialize Polymarket client
host = "https://clob.polymarket.com"
chain_id = 137  # Polygon Mainnet
client = ClobClient(host, key=api_key, chain_id=chain_id)

def fetch_active_markets():
    """Fetch all active markets."""
    markets = []
    next_cursor = None
    while True:
        try:
            # Debug: Print the current cursor
            print(f"Fetching markets with next_cursor: {next_cursor}")
            
            if next_cursor:
                response = client.get_markets(next_cursor=next_cursor)
            else:
                response = client.get_markets()

            # Debug: Print the raw response structure
            # print(f"Response: {response}")

            if 'data' in response:
                markets.extend(response['data'])
            
            # Ensure next_cursor is updated correctly
            next_cursor = response.get("next_cursor", None)
            if not next_cursor or next_cursor == "LTE=":
                break
        except Exception as e:
            print(f"Error while fetching markets: {e}")
            break
    # Filter out active markets
    return [market for market in markets if market.get("active") and not market.get("closed")]


def fetch_order_book(token_id):
    """Fetch order book for a given token ID."""
    try:
        # Fetch the order book
        order_book = client.get_order_book(token_id)
        
        # Ensure the order book has the necessary attributes
        if not hasattr(order_book, 'bids') or not hasattr(order_book, 'asks'):
            print(f"Order book for token_id {token_id} does not have bids or asks.")
            return 0
        
        # Sum up the sizes of bids and asks to calculate total volume
        total_volume = sum(float(order.size) for order in order_book.bids + order_book.asks)
        return total_volume

    except Exception as e:
        print(f"Error fetching order book for token_id {token_id}: {e}")
        return 0


def process_markets(markets):
    """Process markets to calculate trade volumes."""
    processed_data = []
    for market in markets:
        tokens = market.get("tokens", [])
        total_volume = 0
        for token in tokens:
            token_id = token.get("token_id")
            if token_id:
                volume = fetch_order_book(token_id)
                total_volume += volume
        processed_data.append({
            "market_id": market.get("condition_id"),
            "description": market.get("description", "No description available"),
            "category": market.get("category", "Unknown"),
            "total_volume": total_volume,
        })
    return processed_data


def save_to_json(data, output_file="sorted_markets.json"):
    """Save sorted market data to a JSON file."""
    with open(output_file, 'w') as file:
        json.dump(data, file, indent=4)
    print(f"Data saved to {output_file}")

def main():
    print("Fetching active markets...")
    active_markets = fetch_active_markets()
    print(f"Found {len(active_markets)} active markets.")

    print("Processing markets to calculate trade volumes...")
    processed_markets = process_markets(active_markets)

    print("Sorting markets by trade volume...")
    sorted_markets = sorted(processed_markets, key=lambda x: x["total_volume"], reverse=True)

    print("Saving sorted markets to JSON...")
    save_to_json(sorted_markets)

if __name__ == "__main__":
    main()
