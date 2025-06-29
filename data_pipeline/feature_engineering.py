import pandas as pd
import numpy as np
import argparse
import os
import logging


class F1FeatureEngineer:
    """
    Feature engineering class for F1 lap data.
    Creates advanced features for machine learning models.
    """
    
    def __init__(self, debug=False):
        self.debug = debug
        self.logger = logging.getLogger(__name__)
        
    def load_data(self, year, race_name):
        """Load cleaned lap data."""
        input_path = "data/processed/{}/{}/laps_cleaned.csv".format(year, race_name)
        if not os.path.exists(input_path):
            raise FileNotFoundError("Cleaned data not found: {}".format(input_path))
        
        df = pd.read_csv(input_path)
        self.logger.info("Loaded cleaned data: {} rows, {} columns".format(df.shape[0], df.shape[1]))
        return df
    
    def create_lap_dynamics_features(self, df):
        """Create lap dynamics features."""
        self.logger.info("Creating lap dynamics features...")
        
        # Sort by driver and lap number for proper time series operations
        df = df.sort_values(['Driver', 'LapNumber']).reset_index(drop=True)
        
        # Lap delta (time difference to previous lap for same driver)
        df['lap_delta'] = df.groupby('Driver')['LapTime'].diff()
        
        # Position change compared to previous lap
        df['position_change'] = df.groupby('Driver')['Position'].diff() * -1  # Negative for position gain
        
        # Cumulative position changes
        df['cumulative_position_change'] = df.groupby('Driver')['position_change'].cumsum()
        
        # Pit stop indicator (when stint number changes)
        df['pit_stop'] = df.groupby('Driver')['Stint'].diff().fillna(0) > 0
        
        # Cumulative pit stops
        df['pit_stop_count'] = df.groupby('Driver')['pit_stop'].cumsum()
        
        # Laps in current stint
        df['laps_in_stint'] = df.groupby(['Driver', 'Stint']).cumcount() + 1
        
        # Tyre degradation indicators
        df['tyre_age_normalized'] = df['TyreLife'] / df.groupby(['Driver', 'Stint'])['TyreLife'].transform('max')
        
        return df
    
    def create_comparative_features(self, df):
        """Create comparative features against other drivers."""
        self.logger.info("Creating comparative features...")
        
        # Gap to leader (fastest lap time on each lap)
        df['leader_lap_time'] = df.groupby('LapNumber')['LapTime'].transform('min')
        df['gap_to_leader'] = df['LapTime'] - df['leader_lap_time']
        
        # Position-based features
        df = df.sort_values(['LapNumber', 'Position']).reset_index(drop=True)
        
        # Gap to car ahead and behind
        df['car_ahead_time'] = df.groupby('LapNumber')['LapTime'].shift(1)
        df['car_behind_time'] = df.groupby('LapNumber')['LapTime'].shift(-1)
        df['gap_to_ahead'] = df['LapTime'] - df['car_ahead_time']
        df['gap_to_behind'] = df['car_behind_time'] - df['LapTime']
        
        # Percentile ranking within each lap
        df['lap_time_percentile'] = df.groupby('LapNumber')['LapTime'].rank(pct=True)
        
        # Team comparison features
        df['team_avg_lap_time'] = df.groupby(['LapNumber', 'Team'])['LapTime'].transform('mean')
        df['gap_to_teammate'] = df['LapTime'] - df['team_avg_lap_time']
        
        return df
    
    def create_rolling_statistics(self, df):
        """Create rolling window statistics."""
        self.logger.info("Creating rolling statistics...")
        
        # Sort by driver and lap number
        df = df.sort_values(['Driver', 'LapNumber']).reset_index(drop=True)
        
        # Rolling averages for lap times
        for window in [3, 5, 10]:
            df['rolling_avg_lap_time_{}'.format(window)] = (
                df.groupby('Driver')['LapTime']
                .rolling(window=window, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )
            
            # Rolling standard deviation
            df['rolling_std_lap_time_{}'.format(window)] = (
                df.groupby('Driver')['LapTime']
                .rolling(window=window, min_periods=1)
                .std()
                .reset_index(level=0, drop=True)
            )
        
        # Rolling averages for positions
        for window in [3, 5]:
            df['rolling_avg_position_{}'.format(window)] = (
                df.groupby('Driver')['Position']
                .rolling(window=window, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )
        
        # Rolling sector time consistency
        for sector in ['Sector1Time', 'Sector2Time', 'Sector3Time']:
            df['rolling_std_{}_5'.format(sector.lower())] = (
                df.groupby('Driver')[sector]
                .rolling(window=5, min_periods=1)
                .std()
                .reset_index(level=0, drop=True)
            )
        
        return df
    
    def create_driver_context_features(self, df):
        """Create driver and team context features."""
        self.logger.info("Creating driver context features...")
        
        # Driver experience metrics (cumulative over race)
        df['laps_completed'] = df.groupby('Driver').cumcount() + 1
        df['personal_bests_count'] = df.groupby('Driver')['IsPersonalBest'].cumsum()
        
        # Team performance metrics
        df['team_avg_position'] = df.groupby(['LapNumber', 'Team'])['Position'].transform('mean')
        df['team_best_position'] = df.groupby(['LapNumber', 'Team'])['Position'].transform('min')
        
        # Historical performance in race
        df['best_lap_so_far'] = df.groupby('Driver')['LapTime'].cummin()
        df['worst_lap_so_far'] = df.groupby('Driver')['LapTime'].cummax()
        df['avg_lap_so_far'] = df.groupby('Driver')['LapTime'].expanding().mean().reset_index(level=0, drop=True)
        
        # Position statistics
        df['best_position_so_far'] = df.groupby('Driver')['Position'].cummin()
        df['worst_position_so_far'] = df.groupby('Driver')['Position'].cummax()
        
        return df
    
    def create_race_context_features(self, df):
        """Create race context features."""
        self.logger.info("Creating race context features...")
        
        # Race progress
        max_lap = df['LapNumber'].max()
        df['race_progress'] = df['LapNumber'] / max_lap
        
        # Lap phase categorization
        df['race_phase'] = pd.cut(df['race_progress'], 
                                bins=[0, 0.33, 0.66, 1.0], 
                                labels=['early', 'middle', 'late'])
        
        # Encode race phase
        df['race_phase_encoded'] = df['race_phase'].astype('category').cat.codes
        
        # Compound strategy features
        df['compound_age_ratio'] = df['TyreLife'] / (df['TyreLife'].max() + 1)  # Avoid division by zero
        
        # Fresh tyre advantage
        df['fresh_tyre_numeric'] = df['FreshTyre'].astype(int)
        
        return df
    
    def create_speed_analysis_features(self, df):
        """Create speed-based features."""
        self.logger.info("Creating speed analysis features...")
        
        # Speed consistency
        speed_columns = ['SpeedI1', 'SpeedI2', 'SpeedFL', 'SpeedST']
        df['speed_consistency'] = df[speed_columns].std(axis=1)
        df['max_speed'] = df[speed_columns].max(axis=1)
        df['min_speed'] = df[speed_columns].min(axis=1)
        df['speed_range'] = df['max_speed'] - df['min_speed']
        
        # Speed ratios
        df['speed_i1_to_fl_ratio'] = df['SpeedI1'] / (df['SpeedFL'] + 1)  # Avoid division by zero
        df['speed_improvement'] = df['SpeedFL'] - df['SpeedI1']
        
        return df
    
    def create_interaction_features(self, df):
        """Create interaction features."""
        self.logger.info("Creating interaction features...")
        
        # Driver-Team interactions
        df['driver_team_combo'] = df['Driver'].astype(str) + '_' + df['Team'].astype(str)
        df['driver_team_encoded'] = pd.Categorical(df['driver_team_combo']).codes
        
        # Compound-TyreLife interaction
        df['compound_life_interaction'] = df['Compound'] * df['TyreLife']
        
        # Position-Stint interaction (strategy impact)
        df['position_stint_interaction'] = df['Position'] * df['Stint']
        
        return df
    
    def validate_features(self, df):
        """Validate and clean engineered features."""
        self.logger.info("Validating engineered features...")
        
        # Replace infinite values with NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Check for high percentage of NaN values
        nan_percentage = df.isnull().mean()
        high_nan_features = nan_percentage[nan_percentage > 0.5].index.tolist()
        
        if high_nan_features:
            self.logger.warning("Features with >50% NaN values: {}".format(high_nan_features))
        
        # Fill remaining NaN values with appropriate defaults
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(0)
        
        return df
    
    def engineer_features(self, year, race_name):
        """Main feature engineering pipeline."""
        self.logger.info("Starting feature engineering for {} {}".format(year, race_name))
        
        # Load data
        df = self.load_data(year, race_name)
        original_features = df.shape[1]
        
        # Apply feature engineering steps
        df = self.create_lap_dynamics_features(df)
        df = self.create_comparative_features(df)
        df = self.create_rolling_statistics(df)
        df = self.create_driver_context_features(df)
        df = self.create_race_context_features(df)
        df = self.create_speed_analysis_features(df)
        df = self.create_interaction_features(df)
        df = self.validate_features(df)
        
        new_features = df.shape[1] - original_features
        self.logger.info("Feature engineering complete. Added {} new features.".format(new_features))
        self.logger.info("Final dataset: {} rows, {} columns".format(df.shape[0], df.shape[1]))
        
        return df
    
    def save_features(self, df, year, race_name):
        """Save engineered features to CSV."""
        output_dir = "data/processed/{}/{}".format(year, race_name)
        output_path = "{}/laps_features.csv".format(output_dir)
        
        os.makedirs(output_dir, exist_ok=True)
        df.to_csv(output_path, index=False)
        
        self.logger.info("Engineered features saved to {}".format(output_path))
        return output_path


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Engineer features from cleaned F1 lap data.")
    parser.add_argument("--year", type=int, default=2024, help="Race year")
    parser.add_argument("--race", type=str, default="Monaco Grand Prix", help="Race name")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='[%(levelname)s] %(message)s'
    )
    
    # Initialize feature engineer
    engineer = F1FeatureEngineer(debug=args.debug)
    
    try:
        # Engineer features
        df_features = engineer.engineer_features(args.year, args.race)
        
        # Save results
        output_path = engineer.save_features(df_features, args.year, args.race)
        
        print("\n✅ Feature engineering completed successfully!")
        print("📊 Output saved to: {}".format(output_path))
        print("📈 Final dataset shape: {}".format(df_features.shape))
        
        # Display feature summary
        print("\n📋 Feature Summary:")
        print("   Total features: {}".format(df_features.shape[1]))
        print("   Total laps: {}".format(df_features.shape[0]))
        
        # Show some basic statistics
        print("\n📊 Sample Feature Statistics:")
        numeric_cols = df_features.select_dtypes(include=[np.number]).columns
        print(df_features[numeric_cols].describe().iloc[:, :5])  # First 5 numeric columns
        
    except Exception as e:
        logging.error("Feature engineering failed: {}".format(e))
        raise


if __name__ == "__main__":
    main()
