# Data Sources

## FastF1

- **Description:** Python library for accessing Formula 1 telemetry, timing, and results data.
- **Data Types:** Lap times, sector times, car telemetry, weather, session info, etc.
- **Schema:**
  - `Session`: Year, Grand Prix, session type (e.g., race, qualifying)
  - `Lap`: Driver, lap number, lap time, sector times
  - `Telemetry`: Speed, throttle, brake, gear, RPM, position
- **Limitations:**
  - Data availability may vary by season/session
  - Occasional missing or noisy data
  - API rate limits

## Data Quality

- Validate data on extraction
- Handle missing or anomalous values
- Log data issues for review
