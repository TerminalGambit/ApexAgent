import pandas as pd
import argparse
import os

def inspect_cleaned(year, race_name):
    path = f"data/processed/{year}/{race_name}/laps_cleaned.csv"
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return
    df = pd.read_csv(path)
    print("\n--- Columns and Types ---")
    print(df.dtypes)
    print("\n--- First 5 Rows ---")
    print(df.head())
    print("\n--- Missing Values (count) ---")
    print(df.isnull().sum())
    print("\n--- Shape ---")
    print(df.shape)
    print("\n--- Basic Statistics ---")
    print(df.describe(include='all'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect cleaned laps data.")
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--race", type=str, default="Monaco Grand Prix")
    args = parser.parse_args()
    inspect_cleaned(args.year, args.race) 