import fastf1
import argparse
import os
import pandas as pd

def ingest_data(year, race_name, session_type):
    """
    Ingests F1 data for a specific event and session.

    Args:
        year (int): The year of the season.
        race_name (str): The name of the race.
        session_type (str): The type of session (e.g., 'R', 'Q', 'FP1').
    """
    # Load the session
    session = fastf1.get_session(year, race_name, session_type)
    session.load()

    # Load the laps
    laps = session.laps
    print(laps)

    # Create directory to store raw data
    output_dir = f"data/raw/{year}/{race_name}"
    os.makedirs(output_dir, exist_ok=True)

    # Save laps to CSV
    laps.to_csv(f"{output_dir}/laps.csv", index=False)
    print(f"Saved lap data to {output_dir}/laps.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest F1 data.")
    parser.add_argument("--year", type=int, default=2024, help="The year of the season.")
    parser.add_argument("--race", type=str, default="Monaco Grand Prix", help="The name of the race.")
    parser.add_argument("--session", type=str, default="R", help="The type of session (R, Q, FP1, etc.)")
    args = parser.parse_args()

    ingest_data(args.year, args.race, args.session) 