#!/usr/bin/env python3

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os

def parse_args():
    parser = argparse.ArgumentParser(
        description="Check time-series data in a Parquet file."
    )
    parser.add_argument("--parquet", required=True,
                        help="Path to the Parquet file.")
    parser.add_argument("--start_ts", type=int, default=None,
                        help="(Optional) Expected start timestamp.")
    parser.add_argument("--end_ts", type=int, default=None,
                        help="(Optional) Expected end timestamp.")
    parser.add_argument("--fidelity_sec", type=int, default=None,
                        help="(Optional) Expected interval (fidelity) in seconds.")
    parser.add_argument("--duration_sec", type=int, default=None,
                        help="(Optional) Expected duration in seconds.")
    parser.add_argument("--plot", action="store_true",
                        help="If set, plots the price vs. timestamp.")
    return parser.parse_args()

def main():
    args = parse_args()
    parquet_path = args.parquet

    if not os.path.exists(parquet_path):
        print(f"Error: File '{parquet_path}' does not exist.")
        return

    # 1) Read the Parquet file
    print(f"\nReading Parquet file: {parquet_path}")
    df = pd.read_parquet(parquet_path)

    # 2) Basic info
    print("\nDataFrame Summary:")
    print(df.info())

    # 3) Check columns exist
    required_cols = ["timestamp", "price"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        print(f"\nError: Missing required columns: {missing_cols}")
        return

    # 4) Sort by timestamp (in case it's not sorted already)
    df.sort_values(by="timestamp", inplace=True)
    
    # 5) Print min and max timestamps
    min_ts = df["timestamp"].min()
    max_ts = df["timestamp"].max()
    print(f"\nMin timestamp in the data: {min_ts}")
    print(f"Max timestamp in the data: {max_ts}")

    # If user has provided expected start/end, compare
    if args.start_ts is not None:
        print(f"Expected start_ts: {args.start_ts}")
        if min_ts > args.start_ts:
            print("Warning: The earliest data timestamp is after the expected start!")
        else:
            print("Earliest data timestamp <= expected start, looks good.")
    if args.end_ts is not None:
        print(f"Expected end_ts: {args.end_ts}")
        if max_ts < args.end_ts:
            print("Warning: The latest data timestamp is before the expected end!")
        else:
            print("Latest data timestamp >= expected end, looks good.")

    # 6) Check number of rows vs. expected
    row_count = len(df)
    print(f"\nNumber of rows in DataFrame: {row_count}")
    if args.fidelity_sec and args.duration_sec:
        # estimate how many points we might expect
        expected_points = args.duration_sec / args.fidelity_sec
        print(f"Expected ~ {expected_points:.2f} data points "
              f"for a duration of {args.duration_sec} seconds at "
              f"{args.fidelity_sec}-second intervals.")
        diff = row_count - expected_points
        print(f"Actual - Expected = {diff:.2f} (positive => more data than expected, "
              "negative => less data than expected).")

    # 7) Check time deltas between consecutive rows
    #    to see if they match fidelity or if there are big gaps
    df["delta"] = df["timestamp"].diff()
    # dropna() to remove the first row's NaN difference
    differences = df["delta"].dropna()
    if not differences.empty:
        print("\nTimestamp differences summary:")
        print(differences.describe())

        # If user gave a fidelity, let's check how often we see that interval
        if args.fidelity_sec:
            typical_fidelity_rows = sum(differences == args.fidelity_sec)
            print(f"Rows with EXACT difference == {args.fidelity_sec} sec: {typical_fidelity_rows}")
            # You could do a tolerance check if you want, e.g. ±2 seconds
    else:
        print("\nUnable to compute timestamp differences (only one row?).")

    # 8) Plot if requested
    # if args.plot:
    print("\nPlotting data (timestamp vs. price)...")
    plt.figure(figsize=(10, 4))
    plt.plot(df["timestamp"], df["price"], marker='o', linestyle='-')
    plt.xlabel("Timestamp (Unix)")
    plt.ylabel("Price")
    plt.title("Time series data")
    plt.tight_layout()
    plt.show()

    print("\nAll checks done.")

if __name__ == "__main__":
    main()
