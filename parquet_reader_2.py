#!/usr/bin/env python3

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
import ast

def parse_args():
    parser = argparse.ArgumentParser(
        description="Check aggregated time-series data in a Parquet file where each row has arrays for timestamps/prices."
    )
    parser.add_argument("--parquet", required=True,
                        help="Path to the Parquet file.")
    return parser.parse_args()

def main():
    args = parse_args()
    parquet_path = args.parquet

    if not os.path.exists(parquet_path):
        print(f"Error: File '{parquet_path}' does not exist.")
        return

    print(f"Reading Parquet file: {parquet_path}")
    df = pd.read_parquet(parquet_path)

    print("DataFrame info:")
    print(df.info())

    # Expect columns: ["market_id", "market_name", "token_id", "timestamp", "price"]
    # Convert from string if needed
    for idx in df.index:
        if isinstance(df.at[idx, "timestamp"], str):
            try:
                df.at[idx, "timestamp"] = json.loads(df.at[idx, "timestamp"])
            except json.JSONDecodeError:
                df.at[idx, "timestamp"] = ast.literal_eval(df.at[idx, "timestamp"])

        if isinstance(df.at[idx, "price"], str):
            try:
                df.at[idx, "price"] = json.loads(df.at[idx, "price"])
            except json.JSONDecodeError:
                df.at[idx, "price"] = ast.literal_eval(df.at[idx, "price"])

    print("\nAfter potential string-to-list conversion:")
    for idx, row in df.iterrows():
        timestamps = row["timestamp"]
        prices = row["price"]
        print(f"Row {idx} -> timestamps type: {type(timestamps)}, prices type: {type(prices)}")

    # Now your code that checks or plots can go here...

    print("\nDone.")

if __name__ == "__main__":
    main()
