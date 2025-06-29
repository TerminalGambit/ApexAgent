import pandas as pd
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')


class DataLeakageInvestigator:
    """Investigate and fix potential data leakage in features."""
    
    def __init__(self, year, race_name):
        self.year = year
        self.race_name = race_name
        self.data_dir = "data/processed/{}/{}/".format(year, race_name)
        
    def load_data(self):
        """Load the engineered features dataset."""
        print("Loading engineered features dataset...")
        input_path = "data/processed/{}/{}/laps_features.csv".format(self.year, self.race_name)
        self.df = pd.read_csv(input_path)
        print("Loaded dataset with {} rows and {} columns".format(self.df.shape[0], self.df.shape[1]))
        
    def identify_suspicious_features(self):
        """Identify features that might cause data leakage."""
        print("\\nIdentifying potentially leaky features...")
        
        target = 'LapTime'
        numeric_df = self.df.select_dtypes(include=[np.number])
        
        # Calculate correlations with target
        correlations = numeric_df.corr()[target].abs().sort_values(ascending=False)
        
        # Features with perfect or near-perfect correlation (excluding target itself)
        suspicious_features = correlations[(correlations > 0.99) & (correlations.index != target)]
        
        print("Features with >99% correlation to LapTime:")
        for feature, corr in suspicious_features.items():
            print("  {}: {:.6f}".format(feature, corr))
            
        # Analyze specific suspicious patterns
        leaky_patterns = {
            'time_based': [col for col in self.df.columns if 'time' in col.lower() and col != target],
            'current_lap_info': [col for col in self.df.columns if any(x in col.lower() for x in ['ahead', 'behind', 'leader'])],
            'team_averages': [col for col in self.df.columns if 'team_avg' in col.lower()],
            'best_so_far': [col for col in self.df.columns if 'best' in col.lower() and 'so_far' in col.lower()]
        }
        
        print("\\nSuspicious feature patterns:")
        for pattern, features in leaky_patterns.items():
            if features:
                print("  {}: {}".format(pattern, len(features)))
                for f in features[:3]:  # Show first 3
                    print("    - {}".format(f))
                if len(features) > 3:
                    print("    ... and {} more".format(len(features) - 3))
        
        return suspicious_features, leaky_patterns
    
    def create_clean_feature_set(self):
        """Create a cleaned feature set without data leakage."""
        print("\\nCreating clean feature set...")
        
        # Features to definitely remove (likely leaky)
        features_to_remove = [
            # Time-based features that include current lap info
            'leader_lap_time',      # Same as min LapTime for current lap
            'car_ahead_time',       # LapTime of car ahead (sorted by current lap performance)
            'car_behind_time',      # LapTime of car behind
            'team_avg_lap_time',    # Includes current lap
            'best_lap_so_far',      # Cumulative minimum includes current lap
            'worst_lap_so_far',     # Cumulative maximum includes current lap
            'avg_lap_so_far',       # Expanding mean includes current lap
            
            # Gap features that depend on current lap sorting
            'gap_to_ahead',
            'gap_to_behind',
            'gap_to_leader',
            'gap_to_teammate',
            
            # Features that might include current lap in calculation
            'lap_time_percentile',  # Ranking within current lap
        ]
        
        # Keep only lagged/historical features
        safe_features = []
        for col in self.df.columns:
            if col == 'LapTime':  # Keep target
                safe_features.append(col)
            elif col in features_to_remove:
                continue  # Skip leaky features
            elif any(x in col.lower() for x in ['delta', 'change', 'rolling', 'stint', 'tyre', 'sector', 'speed']):
                safe_features.append(col)  # Keep performance and context features
            elif col in ['DriverNumber', 'Driver', 'Team', 'LapNumber', 'Position', 'Stint', 'Compound', 'TyreLife', 'FreshTyre']:
                safe_features.append(col)  # Keep basic race data
            elif 'progress' in col.lower() or 'phase' in col.lower():
                safe_features.append(col)  # Keep race context
            elif col.startswith('personal_bests') or col.startswith('laps_') or col.startswith('pit_'):
                safe_features.append(col)  # Keep driver history
            else:
                print("  Excluding potentially leaky feature: {}".format(col))
        
        self.df_clean = self.df[safe_features].copy()
        print("Clean dataset: {} features (removed {})".format(
            len(safe_features) - 1, len(self.df.columns) - len(safe_features)))  # -1 for target
        
        return safe_features
    
    def test_model_performance(self):
        """Test model performance with clean vs original features."""
        print("\\nTesting model performance...")
        
        results = {}
        
        # Test with original features (from prepared data)
        train_data = pd.read_csv(os.path.join(self.data_dir, 'train_data.csv'))
        test_data = pd.read_csv(os.path.join(self.data_dir, 'test_data.csv'))
        
        X_train_orig = train_data.drop(columns=['LapTime'])
        y_train_orig = train_data['LapTime']
        X_test_orig = test_data.drop(columns=['LapTime'])
        y_test_orig = test_data['LapTime']
        
        # Train with original features
        lr_orig = LinearRegression()
        lr_orig.fit(X_train_orig, y_train_orig)
        y_pred_orig = lr_orig.predict(X_test_orig)
        
        results['original'] = {
            'rmse': np.sqrt(mean_squared_error(y_test_orig, y_pred_orig)),
            'r2': r2_score(y_test_orig, y_pred_orig),
            'n_features': len(X_train_orig.columns)
        }
        
        # Test with clean features
        # Remove outliers and prepare clean data
        target = 'LapTime'
        valid_mask = (self.df_clean[target] >= 60) & (self.df_clean[target] <= 200)
        df_valid = self.df_clean[valid_mask].copy()
        
        # Encode categorical variables if any
        for col in df_valid.columns:
            if df_valid[col].dtype == 'object' and col != target:
                df_valid[col] = pd.Categorical(df_valid[col]).codes
        
        # Split data
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        
        X_clean = df_valid.drop(columns=[target])
        y_clean = df_valid[target]
        
        X_train_clean, X_test_clean, y_train_clean, y_test_clean = train_test_split(
            X_clean, y_clean, test_size=0.2, random_state=42, shuffle=True
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_clean_scaled = scaler.fit_transform(X_train_clean)
        X_test_clean_scaled = scaler.transform(X_test_clean)
        
        # Train with clean features
        lr_clean = LinearRegression()
        lr_clean.fit(X_train_clean_scaled, y_train_clean)
        y_pred_clean = lr_clean.predict(X_test_clean_scaled)
        
        results['clean'] = {
            'rmse': np.sqrt(mean_squared_error(y_test_clean, y_pred_clean)),
            'r2': r2_score(y_test_clean, y_pred_clean),
            'n_features': len(X_train_clean.columns)
        }
        
        # Print comparison
        print("\\nModel Performance Comparison:")
        print("=" * 50)
        print("{:<15} {:<10} {:<10} {:<12}".format("Dataset", "RMSE", "R²", "Features"))
        print("-" * 50)
        print("{:<15} {:<10.3f} {:<10.3f} {:<12}".format(
            "Original", results['original']['rmse'], results['original']['r2'], results['original']['n_features']))
        print("{:<15} {:<10.3f} {:<10.3f} {:<12}".format(
            "Clean", results['clean']['rmse'], results['clean']['r2'], results['clean']['n_features']))
        
        # Determine if leakage was present
        rmse_ratio = results['clean']['rmse'] / results['original']['rmse']
        print("\\nData Leakage Assessment:")
        if rmse_ratio > 10:
            print("🚨 SIGNIFICANT DATA LEAKAGE detected (RMSE increased {:.1f}x)".format(rmse_ratio))
        elif rmse_ratio > 3:
            print("⚠️  MODERATE DATA LEAKAGE detected (RMSE increased {:.1f}x)".format(rmse_ratio))
        elif rmse_ratio > 1.5:
            print("📊 MINOR DATA LEAKAGE detected (RMSE increased {:.1f}x)".format(rmse_ratio))
        else:
            print("✅ No significant data leakage detected")
        
        return results
    
    def save_clean_dataset(self):
        """Save the clean dataset for future use."""
        print("\\nSaving clean dataset...")
        
        # Remove outliers
        target = 'LapTime'
        valid_mask = (self.df_clean[target] >= 60) & (self.df_clean[target] <= 200)
        df_clean_final = self.df_clean[valid_mask].copy()
        
        # Encode categorical variables
        for col in df_clean_final.columns:
            if df_clean_final[col].dtype == 'object' and col != target:
                df_clean_final[col] = pd.Categorical(df_clean_final[col]).codes
        
        # Save
        output_path = os.path.join(self.data_dir, 'laps_features_clean.csv')
        df_clean_final.to_csv(output_path, index=False)
        
        print("Clean dataset saved to: {}".format(output_path))
        print("Final dataset: {} rows, {} features".format(df_clean_final.shape[0], df_clean_final.shape[1] - 1))
        
        return output_path


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Investigate and fix data leakage in F1 features.")
    parser.add_argument("--year", type=int, default=2024, help="Race year")
    parser.add_argument("--race", type=str, default="Monaco Grand Prix", help="Race name")
    
    args = parser.parse_args()
    
    # Initialize investigator
    investigator = DataLeakageInvestigator(args.year, args.race)
    
    try:
        # Load data
        investigator.load_data()
        
        # Identify suspicious features
        suspicious_features, leaky_patterns = investigator.identify_suspicious_features()
        
        # Create clean feature set
        safe_features = investigator.create_clean_feature_set()
        
        # Test performance
        results = investigator.test_model_performance()
        
        # Save clean dataset
        clean_path = investigator.save_clean_dataset()
        
        print("\\n✅ Data leakage investigation completed!")
        print("📊 Clean dataset ready for advanced modeling")
        
    except Exception as e:
        print("❌ Investigation failed: {}".format(e))
        raise


if __name__ == "__main__":
    main()
