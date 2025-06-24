# Building an F1 Machine Learning Pipeline: Lessons, Challenges, and Insights

Blog 1 – June 2025

## Introduction

This post documents the journey of building an end-to-end Formula 1 data analytics and machine learning pipeline. The goal: create a robust, reproducible, and educational system for extracting, cleaning, analyzing, and reporting on F1 race data. Along the way, we faced real-world data challenges, made design decisions, and iterated on our approach. Here's what we learned.

---

## 1. Project Vision & First Steps

We started with a clear vision: build a modular, responsible, and extensible AI system using real F1 data. The first phase focused on the data pipeline:

- **Extract** raw data from the FastF1 API
- **Store** it in a structured, versioned format
- **Clean** and **engineer** features for modeling
- **Visualize** and **report** insights

**Key lesson:** Start small, but design for extensibility and reproducibility from the beginning.

---

## 2. Data Extraction: Getting Raw F1 Data

We used the [FastF1](https://theoehrly.github.io/Fast-F1/) Python library to fetch race, lap, and telemetry data. The first script (`ingest.py`) saved raw lap data as CSV files, organized by year and race name.

**Challenge:** Ensuring the right Python environment and dependencies. We documented everything in `requirements.txt` for reproducibility.

---

## 3. Data Cleaning & Engineering: From Messy to Model-Ready

Raw F1 data is messy! We found:

- Laps with extremely long times (drivers sitting on the grid, crashes, or red flags)
- Missing or mislabelled drivers/teams (e.g., substitutes, edge cases)
- Inconsistent or missing columns

**What we did:**

- Wrote a cleaning script (`clean_laps.py`) to:
  - Convert all time columns to seconds
  - Drop or fill missing values
  - Encode categorical variables
  - Keep key columns like `DriverNumber` and `TeamName` for robust mapping
- Added debug logging and a `--debug` flag for transparency
- Made the process robust to new drivers, teams, and session changes

**Key lesson:** Always keep original identifiers (like driver numbers and team names) for dynamic mapping and future-proofing.

---

## 4. Data Analysis & Visualization

We built a visualization script (`visualize_laps.py`) to:

- Plot lap time distributions
- Show lap time vs. position (with mean lines)
- Compare average lap times per team
- Track lap time progression for a sample driver

**Challenges and questions:**

- How to handle outliers (e.g., laps > 5 minutes)?
- How to map codes to real names dynamically?
- How to make plots and reports robust to missing or new data?

**Key lesson:** Build hashmaps from the data itself, not from hardcoded lists. Filter out outliers for clearer analysis.

---

## 5. Automated Reporting: From Data to PDF

We automated the creation of a LaTeX and PDF report (`generate_report.py`):

- All plots, captions, and interpretations are included
- Filenames are sanitized for LaTeX compatibility
- After report generation, all intermediate images are deleted to keep the workspace clean

**Key lesson:** Automate everything, but always check for edge cases (like special characters in filenames or missing teams).

---

## 6. Iteration & Debugging: Embrace the Feedback Loop

Throughout the process, we:

- Inspected data at every step
- Added debug prints and logging
- Iterated on cleaning, mapping, and reporting logic
- Handled real-world F1 quirks (crashes, substitutes, missing data)

**Key lesson:** Data science is iterative. Expect surprises, and design your pipeline to be flexible and transparent.

---

## 7. What's Next?

- Add more advanced feature engineering (e.g., stint analysis, tire strategies)
- Build interactive dashboards (e.g., with Streamlit)
- Expand to multi-race or multi-season analysis
- Integrate modeling and prediction

---

## Final Thoughts

Building a real-world ML pipeline is about more than just code. It's about:

- Asking the right questions
- Handling messy, evolving data
- Designing for transparency and reproducibility
- Embracing iteration and feedback

We hope this post helps you on your own data science journey—whether in F1 or any other domain!

---

*Stay tuned for more blog posts as the project evolves.*
