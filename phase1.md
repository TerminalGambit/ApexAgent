# F1-ML

This project is an end-to-end machine learning project using Formula 1 data.

## Phase 1: Data Pipeline

The first phase of this project is to build a data pipeline to ingest data from the [FastF1](https://theoehrly.github.io/Fast-F1/) API.

### Usage

To ingest data, run the `ingest.py` script in the `data_pipeline` directory:

```bash
python3 data_pipeline/ingest.py --year 2024 --race "Monaco Grand Prix" --session R
```

This will save the lap data for the specified event to `data/raw/{year}/{race_name}/laps.csv`.

### Dependencies

The dependencies for this project are listed in the `requirements.txt` file. To install them, run:

```bash
pip install -r requirements.txt
```
