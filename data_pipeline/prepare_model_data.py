import pandas as pd
import numpy as np
import argparse
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')


def prepare_model_data(year, race_name, target_variable='LapTime', test_size=0.2, random_state=42):
    """
    Prepare data for machine learning models.
    
    Args:
        year (int): Race year
        race_name (str): Race name
        target_variable (str): Target variable to predict
        test_size (float): Proportion of data for testing
        random_state (int): Random seed for reproducibility
    """
    print("Starting data preparation for {} {}...".format(year, race_name))
    
    # Load feature data
    # Convert race name to directory name (replace spaces with underscores)
    race_dir_name = race_name.replace(" ", "_")
    input_path = "data/processed/{}/{}/laps_features.csv".format(year, race_dir_name)
    df = pd.read_csv(input_path)
    print("Loaded dataset with {} rows and {} columns".format(df.shape[0], df.shape[1]))
    
    # Create output directory
    output_dir = "data/processed/{}/{}/".format(year, race_dir_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Remove highly correlated features based on previous analysis
    print("\\nRemoving highly correlated features...")
    features_to_remove = [
        # Perfect correlations (keeping the more interpretable ones)
        'laps_completed',  # Same as LapNumber
        'race_progress',   # Same as LapNumber normalized
        'leader_lap_time', # Same as LapTime for leader
        'team_avg_lap_time', # Same as LapTime for team average
        'best_lap_so_far', # Same as LapTime minimum
        'pit_stop_count',  # Same as Stint
        'compound_age_ratio', # Same as TyreLife normalized
        'rolling_avg_position_5', # Keep rolling_avg_position_3
        'rolling_avg_lap_time_10', # Keep rolling_avg_lap_time_5
        'avg_lap_so_far', # Similar to rolling averages
        'worst_position_so_far', # Keep best_position_so_far
        'speed_improvement', # Inverse of speed_i1_to_fl_ratio
    ]
    
    df_clean = df.drop(columns=[col for col in features_to_remove if col in df.columns])
    print("Removed {} highly correlated features".format(len([col for col in features_to_remove if col in df.columns])))
    print("Dataset now has {} columns".format(df_clean.shape[1]))
    
    # Handle missing values
    print("\\nHandling missing values...")
    missing_before = df_clean.isnull().sum().sum()
    
    # Fill numeric missing values with median
    numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if df_clean[col].isnull().sum() > 0:
            median_val = df_clean[col].median()
            df_clean[col].fillna(median_val, inplace=True)
    
    # Fill categorical missing values with mode
    categorical_columns = df_clean.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        if df_clean[col].isnull().sum() > 0:
            mode_val = df_clean[col].mode()[0] if len(df_clean[col].mode()) > 0 else 'Unknown'
            df_clean[col].fillna(mode_val, inplace=True)
    
    missing_after = df_clean.isnull().sum().sum()
    print("Filled {} missing values".format(missing_before - missing_after))
    
    # Encode categorical variables
    print("\\nEncoding categorical variables...")
    categorical_columns = df_clean.select_dtypes(include=['object']).columns
    if len(categorical_columns) > 0:
        label_encoders = {}
        for col in categorical_columns:
            if col != target_variable:  # Don't encode target if it's categorical
                le = LabelEncoder()
                df_clean[col] = le.fit_transform(df_clean[col].astype(str))
                label_encoders[col] = le
        print("Encoded {} categorical columns".format(len(categorical_columns)))
    
    # Prepare features and target
    if target_variable not in df_clean.columns:
        raise ValueError("Target variable '{}' not found in dataset".format(target_variable))
    
    # Remove target from features
    feature_columns = [col for col in df_clean.columns if col != target_variable]
    X = df_clean[feature_columns]
    y = df_clean[target_variable]
    
    print("\\nPrepared {} features for target variable: {}".format(len(feature_columns), target_variable))
    
    # Handle outliers in target variable (remove extreme values)
    if target_variable == 'LapTime':
        # Remove lap times that are too fast (< 60s) or too slow (> 200s for Monaco)
        valid_mask = (y >= 60) & (y <= 200)
        X = X[valid_mask]
        y = y[valid_mask]
        print("Removed {} outlier lap times".format(sum(~valid_mask)))
    
    # Split data
    print("\\nSplitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=True
    )
    
    print("Train set: {} samples".format(len(X_train)))
    print("Test set: {} samples".format(len(X_test)))
    
    # Scale features
    print("\\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrames
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    # Save prepared data
    print("\\nSaving prepared datasets...")
    
    # Save train data
    train_data = X_train_scaled.copy()
    train_data[target_variable] = y_train.values
    train_path = os.path.join(output_dir, 'train_data.csv')
    train_data.to_csv(train_path, index=False)
    
    # Save test data
    test_data = X_test_scaled.copy()
    test_data[target_variable] = y_test.values
    test_path = os.path.join(output_dir, 'test_data.csv')
    test_data.to_csv(test_path, index=False)
    
    # Save feature names
    feature_names = pd.DataFrame({'feature': X_train.columns})
    feature_names_path = os.path.join(output_dir, 'feature_names.csv')
    feature_names.to_csv(feature_names_path, index=False)
    
    # Save data preparation info
    prep_info = {
        'target_variable': target_variable,
        'n_features': len(feature_columns),
        'n_train_samples': len(X_train),
        'n_test_samples': len(X_test),
        'test_size': test_size,
        'random_state': random_state,
        'features_removed': features_to_remove,
        'categorical_columns': list(categorical_columns)
    }
    
    prep_info_df = pd.Series(prep_info).to_frame('value')
    prep_info_path = os.path.join(output_dir, 'preparation_info.csv')
    prep_info_df.to_csv(prep_info_path)
    
    print("\\n✅ Data preparation completed!")
    print("📊 Generated files:")
    print("   - train_data.csv ({} samples)".format(len(X_train)))
    print("   - test_data.csv ({} samples)".format(len(X_test)))
    print("   - feature_names.csv")
    print("   - preparation_info.csv")
    
    # Print summary statistics
    print("\\n📈 Target variable statistics:")
    print("Train - Mean: {:.2f}, Std: {:.2f}, Min: {:.2f}, Max: {:.2f}".format(
        y_train.mean(), y_train.std(), y_train.min(), y_train.max()))
    print("Test  - Mean: {:.2f}, Std: {:.2f}, Min: {:.2f}, Max: {:.2f}".format(
        y_test.mean(), y_test.std(), y_test.min(), y_test.max()))
    
    return {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': list(X_train.columns),
        'scaler': scaler,
        'output_dir': output_dir
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data for machine learning models.")
    parser.add_argument("--year", type=int, default=2024, help="Race year")
    parser.add_argument("--race", type=str, default="Monaco Grand Prix", help="Race name")
    parser.add_argument("--target", type=str, default="LapTime", help="Target variable to predict")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set proportion")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    prepare_model_data(
        year=args.year,
        race_name=args.race,
        target_variable=args.target,
        test_size=args.test_size,
        random_state=args.random_state
    )
