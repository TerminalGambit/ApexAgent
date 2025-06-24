import pandas as pd
import argparse
import os
import numpy as np
import logging

def time_to_seconds(t):
    if pd.isnull(t):
        return np.nan
    try:
        # Try pandas to_timedelta first
        td = pd.to_timedelta(t)
        return td.total_seconds()
    except Exception:
        pass
    try:
        # Fallback: manual parsing
        if isinstance(t, float):
            return t
        if ":" in str(t):
            parts = str(t).split(":")
            if len(parts) == 2:
                m, s = parts
                return float(m) * 60 + float(s)
            elif len(parts) == 3:
                h, m, s = parts
                return float(h) * 3600 + float(m) * 60 + float(s)
        return float(t)
    except Exception as e:
        logging.warning(f"Could not convert time value '{t}' to seconds: {e}")
        return np.nan

def clean_laps(year, race_name, debug=False):
    input_path = f"data/raw/{year}/{race_name}/laps.csv"
    output_dir = f"data/processed/{year}/{race_name}"
    output_path = f"{output_dir}/laps_cleaned.csv"
    if not os.path.exists(input_path):
        logging.error(f"File not found: {input_path}")
        return
    df = pd.read_csv(input_path)
    logging.info(f"Loaded raw data: {df.shape[0]} rows, {df.shape[1]} columns")

    # Columns to keep (add DriverNumber and TeamName if present)
    keep_cols = [
        'DriverNumber', 'Driver', 'Team', 'TeamName', 'LapNumber', 'LapTime', 'Stint',
        'Sector1Time', 'Sector2Time', 'Sector3Time',
        'SpeedI1', 'SpeedI2', 'SpeedFL', 'SpeedST',
        'Compound', 'TyreLife', 'FreshTyre',
        'IsPersonalBest', 'Position'
    ]
    keep_cols = [col for col in keep_cols if col in df.columns]
    missing_cols = [col for col in keep_cols if col not in df.columns]
    if missing_cols:
        logging.warning(f"Missing columns in raw data: {missing_cols}")
    df = df[keep_cols]
    logging.info(f"After column selection: {df.shape[0]} rows, {df.shape[1]} columns")

    # Convert time columns to seconds
    for col in ['LapTime', 'Sector1Time', 'Sector2Time', 'Sector3Time']:
        if col in df.columns:
            df[col] = df[col].apply(time_to_seconds)
    logging.info(f"After time conversion: {df.shape[0]} rows")

    # Drop rows with missing LapTime or Position
    before_drop = df.shape[0]
    df = df.dropna(subset=[col for col in ['LapTime', 'Position'] if col in df.columns])
    logging.info(f"Dropped {before_drop - df.shape[0]} rows with missing LapTime or Position. Remaining: {df.shape[0]}")

    # Fill missing sector times and speeds with median
    for col in ['Sector1Time', 'Sector2Time', 'Sector3Time', 'SpeedI1', 'SpeedI2', 'SpeedFL', 'SpeedST']:
        if col in df.columns:
            median = df[col].median()
            df[col] = df[col].fillna(median)
    logging.info(f"After filling missing sector/speed values: {df.shape[0]} rows")

    # Encode categorical variables
    for col in ['Compound', 'Team', 'Driver']:
        if col in df.columns:
            df[col] = df[col].astype('category').cat.codes
    logging.info(f"After encoding categoricals: {df.shape[0]} rows")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(output_path, index=False)
    logging.info(f"Cleaned data saved to {output_path}")
    if df.shape[0] == 0:
        logging.warning("All rows were dropped during cleaning! Check your data and cleaning logic.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean and prepare laps data.")
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--race", type=str, default="Monaco Grand Prix")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO, format='[%(levelname)s] %(message)s')
    clean_laps(args.year, args.race, debug=args.debug) 