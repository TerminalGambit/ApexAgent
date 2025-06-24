import pandas as pd
import argparse
import os

def analyze_laps(year, race_name):
    path = f"data/raw/{year}/{race_name}/laps.csv"
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze raw laps data.")
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--race", type=str, default="Monaco Grand Prix")
    args = parser.parse_args()
    analyze_laps(args.year, args.race) 