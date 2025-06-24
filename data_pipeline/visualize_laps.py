import pandas as pd
import argparse
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
import re

sns.set(style="whitegrid")

def seconds_to_mmss(seconds):
    m = int(seconds // 60)
    s = seconds % 60
    return f"{m}:{s:05.2f}"

def sanitize_filename(s):
    if not isinstance(s, str):
        s = str(s)
    # Replace special characters with underscores
    return re.sub(r'[^A-Za-z0-9_\-]', '_', s)

def visualize_laps(year, race_name):
    path = f"data/processed/{year}/{race_name}/laps_cleaned.csv"
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return
    df = pd.read_csv(path)

    # Build driver and team hashmaps from the data
    driver_number_to_code = dict(zip(df['DriverNumber'], df['Driver'])) if 'DriverNumber' in df.columns else {}
    driver_code_to_number = dict(zip(df['Driver'], df['DriverNumber'])) if 'DriverNumber' in df.columns else {}
    driver_number_to_name = {num: f"Driver {code}" for num, code in driver_number_to_code.items()}
    if 'Driver' in df.columns and 'DriverNumber' in df.columns:
        # If you have a mapping of code to name, you can update here
        pass  # Placeholder for future mapping

    if 'TeamName' in df.columns:
        team_code_to_name = dict(zip(df['Team'], df['TeamName']))
    else:
        team_code_to_name = {code: f"Team {code}" for code in df['Team'].unique()}

    # Print all unique driver and team codes
    print("DriverNumber to Driver code:")
    print(driver_number_to_code)
    print("Team code to TeamName:")
    print(team_code_to_name)

    # Filter out laps with LapTime > 300 seconds (5 minutes)
    before = df.shape[0]
    df = df[df['LapTime'] <= 300]
    after = df.shape[0]
    if before != after:
        print(f"[INFO] Filtered out {before - after} laps with LapTime > 5 minutes.")

    # Prepare for LaTeX report
    report = {}

    # 1. Lap time distribution (mm:ss)
    plt.figure(figsize=(8, 5))
    sns.histplot(df['LapTime'], bins=30, kde=True)
    plt.title('Lap Time Distribution')
    plt.xlabel('Lap Time (mm:ss)')
    plt.ylabel('Count')
    ticks = plt.xticks()[0]
    plt.xticks(ticks, [seconds_to_mmss(t) for t in ticks])
    plt.tight_layout()
    plot1_path = f"data/processed/{year}/{race_name}/lap_time_distribution.png"
    plt.savefig(plot1_path)
    plt.show()
    report['lap_time_distribution'] = {
        'path': plot1_path,
        'caption': "Lap Time Distribution",
        'interpretation': "Most lap times cluster around the mean, with a few slower laps (possibly due to pit stops or incidents)."
    }

    # 2. Lap time vs. position (all laps, with mean line)
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x='Position', y='LapTime', data=df, alpha=0.5, label='All laps')
    mean_lap = df.groupby('Position')['LapTime'].mean().reset_index()
    plt.plot(mean_lap['Position'], mean_lap['LapTime'], color='red', label='Mean Lap Time', linewidth=2)
    plt.title('Lap Time vs. Position')
    plt.xlabel('Position (lower is better)')
    plt.ylabel('Lap Time (mm:ss)')
    ticks = plt.yticks()[0]
    plt.yticks(ticks, [seconds_to_mmss(t) for t in ticks])
    plt.legend()
    plt.tight_layout()
    plot2_path = f"data/processed/{year}/{race_name}/lap_time_vs_position.png"
    plt.savefig(plot2_path)
    plt.show()
    report['lap_time_vs_position'] = {
        'path': plot2_path,
        'caption': "Lap Time vs. Position",
        'interpretation': "Drivers in better positions (closer to 1) tend to have faster lap times, as expected. The red line shows the mean lap time for each position."
    }

    # 3. Average lap time per team (team names)
    plt.figure(figsize=(10, 5))
    avg_lap_time = df.groupby('Team')['LapTime'].mean().reset_index()
    avg_lap_time['TeamName'] = avg_lap_time['Team'].map(team_code_to_name)
    sns.barplot(x='TeamName', y='LapTime', data=avg_lap_time)
    plt.title('Average Lap Time per Team')
    plt.xlabel('Team')
    plt.ylabel('Average Lap Time (mm:ss)')
    ticks = plt.yticks()[0]
    plt.yticks(ticks, [seconds_to_mmss(t) for t in ticks])
    plt.xticks(rotation=45)
    plt.tight_layout()
    plot3_path = f"data/processed/{year}/{race_name}/avg_lap_time_per_team.png"
    plt.savefig(plot3_path)
    plt.show()
    report['avg_lap_time_per_team'] = {
        'path': plot3_path,
        'caption': "Average Lap Time per Team",
        'interpretation': "Teams with lower average lap times are generally more competitive. Note: Some teams may be missing due to incidents."
    }

    # 4. Lap time progression for a sample driver (driver number and code)
    sample_driver_number = df['DriverNumber'].unique()[0] if 'DriverNumber' in df.columns else None
    if sample_driver_number is not None:
        driver_laps = df[df['DriverNumber'] == sample_driver_number]
        driver_code = driver_number_to_code.get(sample_driver_number, str(sample_driver_number))
        driver_label = f"{driver_code}_num{sample_driver_number}"
    else:
        driver_laps = df.iloc[0:0]
        driver_label = "Unknown"
    safe_driver_label = sanitize_filename(driver_label)
    plt.figure(figsize=(10, 5))
    plt.plot(driver_laps['LapNumber'], driver_laps['LapTime'], marker='o')
    plt.title(f'Lap Time Progression for {driver_label.replace("_", " ")}')
    plt.xlabel('Lap Number')
    plt.ylabel('Lap Time (mm:ss)')
    ticks = plt.yticks()[0]
    plt.yticks(ticks, [seconds_to_mmss(t) for t in ticks])
    plt.tight_layout()
    plot4_path = f"data/processed/{year}/{race_name}/lap_time_progression_driver_{safe_driver_label}.png"
    plt.savefig(plot4_path)
    plt.show()
    report['lap_time_progression'] = {
        'path': plot4_path,
        'caption': f"Lap Time Progression for {driver_label.replace('_', ' ')}",
        'interpretation': f"This shows how lap times change for {driver_label.replace('_', ' ')} over the race. Dips may indicate pit stops or faster laps."
    }

    # Save report dictionary for LaTeX report generation
    report_path = f"data/processed/{year}/{race_name}/report_data.pkl"
    with open(report_path, 'wb') as f:
        pickle.dump(report, f)
    print(f"[INFO] Report data saved to {report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize cleaned laps data.")
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--race", type=str, default="Monaco Grand Prix")
    args = parser.parse_args()
    visualize_laps(args.year, args.race) 