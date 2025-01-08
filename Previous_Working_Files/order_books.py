import os
import logging
import pandas as pd
from py_clob_client.client import ClobClient

# Initialize the ClobClient
api_key = os.getenv('API_KEY')  # Ensure your API key is set in the environment
host = "https://clob.polymarket.com"
chain_id = 137  # Polygon Mainnet
client = ClobClient(host, key=api_key, chain_id=chain_id)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_markets_from_parquet(parquet_file):
    """
    Load the market data from a Parquet file.
    
    Args:
        parquet_file (str): Path to the Parquet file.
    
    Returns:
        pd.DataFrame: A DataFrame with market data.
    """
    try:
        markets = pd.read_parquet(parquet_file)
        logging.info(f"Loaded {len(markets)} markets from {parquet_file}")
        return markets
    except Exception as e:
        logging.error(f"Error loading Parquet file: {e}")
        raise


def fetch_order_book_for_market(token_id):
    """
    Fetch the order book for a given token ID.
    
    Args:
        token_id (str): The token ID of the market.
    
    Returns:
        dict: The fetched order book.
    """
    try:
        return client.get_order_book(token_id)
    except Exception as e:
        logging.error(f"Failed to fetch order book for token_id {token_id}: {e}")
        return None


def process_and_save_order_books(markets_df, output_dir="./order_books"):
    """
    Fetch and save order books for all markets.
    
    Args:
        markets_df (pd.DataFrame): DataFrame containing markets data.
        output_dir (str): Directory to save order book data.
    """
    os.makedirs(output_dir, exist_ok=True)

    for _, row in markets_df.iterrows():
        market_name = row.get("market_name", "unknown_market")
        tokens = row.get("tokens", [])
        
        for token in tokens:
            token_id = token.get("token_id")
            if not token_id:
                logging.warning(f"No token ID found for market: {market_name}")
                continue

            logging.info(f"Fetching order book for market: {market_name}, token ID: {token_id}")
            order_book = fetch_order_book_for_market(token_id)

            if not order_book or not hasattr(order_book, "bids") or not hasattr(order_book, "asks"):
                logging.warning(f"Order book is empty or malformed for token_id: {token_id}")
                continue

            # Process the order book into a DataFrame
            book_data = []
            for side, orders in [("asks", order_book.asks), ("bids", order_book.bids)]:
                for order in orders:
                    book_data.append({
                        "market_name": market_name,
                        "token_id": token_id,
                        "price": float(order.price),
                        "size": float(order.size),
                        "side": side
                    })

            if book_data:
                df = pd.DataFrame(book_data)
                file_name = f"{market_name}_{token_id}.csv"
                output_path = os.path.join(output_dir, file_name)
                df.to_csv(output_path, index=False)
                logging.info(f"Saved order book for {market_name} ({token_id}) to {output_path}")
            else:
                logging.warning(f"No data found in order book for token_id: {token_id}")


def main():
    # Path to the Parquet file
    parquet_file = "/Users/ru/Polymarket/time_series_data.parquet"  # Update this path as needed
    
    # Load the markets data from the Parquet file
    markets_df = load_markets_from_parquet(parquet_file)
    
    # Process and save order books for all markets
    process_and_save_order_books(markets_df)
    
    # Output an example of one order book
    if not markets_df.empty:
        example_token = markets_df.iloc[0]["tokens"][0]["token_id"]
        example_order_book = fetch_order_book_for_market(example_token)
        print("\nExample Order Book:")
        print(example_order_book)


if __name__ == "__main__":
    main()
