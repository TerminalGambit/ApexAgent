# Phase 2: Feature Store & Representation

## Overview

Feature engineering transforms raw data into meaningful inputs for machine learning models. A feature store abstracts and manages these features for reuse and versioning.

## Current Feature Implementation (v1.0)

### Raw Features (18 columns)
Original features from the cleaned dataset:
- **Driver Context:** DriverNumber, Driver, Team
- **Lap Timing:** LapNumber, LapTime, Sector1Time, Sector2Time, Sector3Time
- **Speed Data:** SpeedI1, SpeedI2, SpeedFL, SpeedST
- **Tire Strategy:** Compound, TyreLife, FreshTyre
- **Performance:** Stint, IsPersonalBest, Position

### Engineered Features (51 new columns)

#### 1. Lap Dynamics Features
- **lap_delta:** Time difference to previous lap for same driver
- **position_change:** Position change compared to previous lap (negative = position gain)
- **cumulative_position_change:** Running total of position changes
- **pit_stop:** Boolean indicator when stint number changes
- **pit_stop_count:** Cumulative pit stops for driver
- **laps_in_stint:** Number of laps since last pit stop
- **tyre_age_normalized:** Tyre life normalized within stint (0-1)

#### 2. Comparative Features
- **leader_lap_time:** Fastest lap time on each lap number
- **gap_to_leader:** Time difference to race leader
- **car_ahead_time, car_behind_time:** Lap times of cars directly ahead/behind
- **gap_to_ahead, gap_to_behind:** Time gaps to adjacent cars
- **lap_time_percentile:** Driver's percentile ranking within each lap
- **team_avg_lap_time:** Average team performance per lap
- **gap_to_teammate:** Performance relative to team average

#### 3. Rolling Statistics (Windows: 3, 5, 10 laps)
- **rolling_avg_lap_time_{window}:** Moving average of lap times
- **rolling_std_lap_time_{window}:** Moving standard deviation (consistency)
- **rolling_avg_position_{3,5}:** Moving average positions
- **rolling_std_{sector}_5:** Sector time consistency over 5 laps

#### 4. Driver Context Features
- **laps_completed:** Cumulative laps completed in race
- **personal_bests_count:** Running count of personal best laps
- **best_lap_so_far, worst_lap_so_far:** Extremes up to current lap
- **avg_lap_so_far:** Expanding average lap time
- **best_position_so_far, worst_position_so_far:** Position extremes

#### 5. Race Context Features
- **race_progress:** Normalized race completion (0-1)
- **race_phase:** Categorical (early/middle/late) and encoded versions
- **compound_age_ratio:** Normalized tyre age across race
- **fresh_tyre_numeric:** Binary fresh tyre indicator

#### 6. Speed Analysis Features
- **speed_consistency:** Standard deviation across speed traps
- **max_speed, min_speed, speed_range:** Speed trap statistics
- **speed_i1_to_fl_ratio:** Entry to finish line speed ratio
- **speed_improvement:** Speed gain from sector 1 to finish line

#### 7. Interaction Features
- **driver_team_combo:** String combination for categorical encoding
- **driver_team_encoded:** Numeric encoding of driver-team pairs
- **compound_life_interaction:** Tyre compound × tyre life
- **position_stint_interaction:** Position × stint strategy impact

## Feature Engineering Pipeline

### Implementation
- **Script:** `data_pipeline/feature_engineering.py`
- **Input:** `laps_cleaned.csv` (18 features)
- **Output:** `laps_features.csv` (69 features)
- **Added Features:** 51 engineered features

### Usage
```bash
python3 data_pipeline/feature_engineering.py --year 2024 --race "Monaco Grand Prix"
```

### Validation Steps
1. Replace infinite values with NaN
2. Identify features with >50% missing values
3. Fill remaining NaN with appropriate defaults (0 for numeric)
4. Ensure data type consistency

## Versioning Strategies

- **v1.0:** Initial comprehensive feature set (69 features total)
- Store features with version tags in file naming
- Document feature changes in this file
- Track feature importance and selection in model experiments

## Feature Selection & Importance

- Use statistical tests and model-based methods (e.g., feature importances from tree models)
- Monitor feature correlation and multicollinearity
- Regularly review and update feature sets for optimal performance
- Consider domain expertise for F1-specific feature engineering

## Principles

- **Reusability:** Features are defined once and reused across models
- **Versioning:** Track changes to features for reproducibility
- **Validation:** Ensure data quality and handle edge cases
- **Documentation:** Clear naming and comprehensive documentation

## Outcome

A flexible, maintainable feature engineering process that:
- Increased feature space from 18 to 69 columns (283% increase)
- Captures temporal dynamics, competitive context, and strategic elements
- Provides robust foundation for machine learning model development
- Accelerates experimentation and ensures consistency across models
