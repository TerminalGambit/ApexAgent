#!/usr/bin/env python3
"""
LSTM Integration Test for F1-ML Platform

This script integrates the new LSTM model with your existing F1 data pipeline
and compares performance against your current baseline models.

Usage:
    python lstm_integration_test.py
"""

import sys
import os
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Add paths to import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))

try:
    from models.advanced.deep_learning.lstm_model import F1LSTMPredictor
    print("✅ Successfully imported LSTM model")
except ImportError as e:
    print(f"❌ Failed to import LSTM model: {e}")
    sys.exit(1)

class F1ModelComparison:
    """
    Compare LSTM model against existing baseline models.
    """
    
    def __init__(self, data_dir="../../data/processed/2024/Monaco Grand Prix/"):
        """
        Initialize the comparison framework.
        
        Args:
            data_dir: Path to processed F1 data
        """
        self.data_dir = Path(data_dir)
        self.results = {}
        
        print(f"🏎️ F1 LSTM Model Integration Test")
        print(f"📁 Data directory: {self.data_dir}")
        
    def load_data(self):
        """Load F1 lap data for testing."""
        print("\n📊 Loading F1 data...")
        
        try:
            # Try to load your actual processed data
            lap_features_file = self.data_dir / "laps_features.csv"
            train_file = self.data_dir / "train_data.csv"
            test_file = self.data_dir / "test_data.csv"
            
            if lap_features_file.exists():
                self.lap_data = pd.read_csv(lap_features_file)
                print(f"✅ Loaded {len(self.lap_data)} lap records from {lap_features_file}")
            else:
                print(f"⚠️ Lap features file not found at {lap_features_file}")
                self.lap_data = self.create_sample_data()
                
            if train_file.exists() and test_file.exists():
                self.train_data = pd.read_csv(train_file)
                self.test_data = pd.read_csv(test_file)
                print(f"✅ Loaded train ({len(self.train_data)}) and test ({len(self.test_data)}) splits")
            else:
                print("⚠️ Train/test files not found, creating split...")
                self.create_train_test_split()
                
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            print("🔄 Creating sample data for demo...")
            self.lap_data = self.create_sample_data()
            self.create_train_test_split()
            
    def create_sample_data(self):
        """Create sample F1 data with realistic structure."""
        print("🎭 Creating sample F1 data...")
        
        np.random.seed(42)
        n_laps = 2000
        
        # Monaco-like lap times (around 74-78 seconds)
        base_lap_times = np.random.normal(76, 2, n_laps)
        
        # Driver list
        drivers = ['HAM', 'VER', 'LEC', 'NOR', 'SAI', 'RUS', 'PER', 'ALO', 'OCO', 'GAS']
        teams = ['Mercedes', 'Red Bull', 'Ferrari', 'McLaren', 'Ferrari', 'Mercedes', 'Red Bull', 'Aston Martin', 'Alpine', 'Alpine']
        
        # Create realistic lap progression
        lap_numbers = []
        driver_names = []
        team_names = []
        
        for i, driver in enumerate(drivers):
            n_driver_laps = n_laps // len(drivers)
            lap_numbers.extend(range(1, n_driver_laps + 1))
            driver_names.extend([driver] * n_driver_laps)
            team_names.extend([teams[i]] * n_driver_laps)
        
        # Adjust array sizes
        lap_numbers = lap_numbers[:n_laps]
        driver_names = driver_names[:n_laps]
        team_names = team_names[:n_laps]
        
        # Create features
        data = pd.DataFrame({
            'LapTime': base_lap_times,
            'Driver': driver_names,
            'Team': team_names,
            'LapNumber': lap_numbers,
            'Position': np.random.randint(1, 21, n_laps),
            'Sector1Time': base_lap_times * 0.31 + np.random.normal(0, 0.5, n_laps),
            'Sector2Time': base_lap_times * 0.46 + np.random.normal(0, 0.8, n_laps),
            'Sector3Time': base_lap_times * 0.23 + np.random.normal(0, 0.3, n_laps),
            'TyreLife': np.random.randint(1, 40, n_laps),
            'TrackTemp': np.random.normal(35, 5, n_laps),
            'AirTemp': np.random.normal(25, 3, n_laps)
        })
        
        # Add engineered features
        data = self.add_engineered_features(data)
        
        return data
        
    def add_engineered_features(self, data):
        """Add engineering features similar to your existing pipeline."""
        print("🔧 Adding engineered features...")
        
        data = data.sort_values(['Driver', 'LapNumber'])
        
        # Rolling averages
        data['rolling_avg_3'] = data.groupby('Driver')['LapTime'].rolling(3, min_periods=1).mean().reset_index(0, drop=True)
        data['rolling_avg_5'] = data.groupby('Driver')['LapTime'].rolling(5, min_periods=1).mean().reset_index(0, drop=True)
        data['rolling_avg_10'] = data.groupby('Driver')['LapTime'].rolling(10, min_periods=1).mean().reset_index(0, drop=True)
        
        # Best lap so far
        data['best_lap_so_far'] = data.groupby('Driver')['LapTime'].cummin()
        
        # Team averages
        data['team_avg_lap_time'] = data.groupby('Team')['LapTime'].transform('mean')
        
        # Sector deltas
        data['sector1_delta'] = data['Sector1Time'] - data.groupby('Driver')['Sector1Time'].transform('mean')
        data['sector2_delta'] = data['Sector2Time'] - data.groupby('Driver')['Sector2Time'].transform('mean')
        data['sector3_delta'] = data['Sector3Time'] - data.groupby('Driver')['Sector3Time'].transform('mean')
        
        # Performance metrics
        data['lap_time_delta'] = data['LapTime'] - data['best_lap_so_far']
        data['position_delta'] = data['Position'] - data.groupby('Driver')['Position'].shift(1)
        
        # Fill NaN values
        data = data.fillna(method='bfill').fillna(method='ffill').fillna(0)
        
        return data
        
    def create_train_test_split(self):
        """Create train/test split."""
        print("🔀 Creating train/test split...")
        
        # Use time-based split (first 80% for training)
        split_point = int(0.8 * len(self.lap_data))
        
        self.train_data = self.lap_data.iloc[:split_point].copy()
        self.test_data = self.lap_data.iloc[split_point:].copy()
        
        print(f"✅ Train: {len(self.train_data)} samples, Test: {len(self.test_data)} samples")
        
    def load_baseline_models(self):
        """Load existing baseline models for comparison."""
        print("\n🤖 Loading baseline models...")
        
        models_dir = Path("../trained/")
        self.baseline_models = {}
        
        # Try to load your existing models
        model_files = {
            'ElasticNet': 'advanced_ElasticNet.joblib',
            'XGBoost': 'baseline_XGBoost.joblib',
            'RandomForest': 'baseline_RandomForest.joblib'
        }
        
        for model_name, filename in model_files.items():
            model_path = models_dir / filename
            if model_path.exists():
                try:
                    loaded_model = joblib.load(model_path)
                    # Wrap in proper structure for evaluation
                    self.baseline_models[model_name] = {
                        'model': loaded_model,
                        'scaler': None,  # Assume already scaled in training
                        'features': None  # Will be determined from data
                    }
                    print(f"✅ Loaded {model_name} model")
                except Exception as e:
                    print(f"⚠️ Failed to load {model_name}: {e}")
            else:
                print(f"⚠️ Model file not found: {model_path}")
                
        if not self.baseline_models:
            print("⚠️ No baseline models found, will create simple baselines for comparison")
            self.create_simple_baselines()
            
    def create_simple_baselines(self):
        """Create simple baseline models for comparison."""
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import ElasticNet
        from sklearn.preprocessing import StandardScaler
        
        print("🔨 Creating simple baseline models...")
        
        # Prepare features
        feature_cols = [col for col in self.train_data.columns 
                       if col not in ['LapTime', 'Driver', 'Team'] and 
                       self.train_data[col].dtype in ['int64', 'float64']]
        
        X_train = self.train_data[feature_cols].fillna(0)
        y_train = self.train_data['LapTime']
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Train simple models
        self.baseline_models = {}
        
        # Random Forest
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        self.baseline_models['RandomForest'] = {'model': rf, 'scaler': None, 'features': feature_cols}
        
        # ElasticNet
        en = ElasticNet(random_state=42)
        en.fit(X_train_scaled, y_train)
        self.baseline_models['ElasticNet'] = {'model': en, 'scaler': scaler, 'features': feature_cols}
        
        print(f"✅ Created {len(self.baseline_models)} baseline models")
        
    def train_lstm_model(self):
        """Train the LSTM model."""
        print("\n🧠 Training LSTM model...")
        
        # Initialize LSTM predictor
        self.lstm_predictor = F1LSTMPredictor(
            sequence_length=8,      # Use last 8 laps to predict next one
            hidden_size=64,         # Moderate size for faster training
            num_layers=2,           # 2-layer LSTM
            dropout=0.2,            # Regularization
            use_attention=True      # Enable attention mechanism
        )
        
        # Prepare data
        train_loader, val_loader = self.lstm_predictor.prepare_data(self.train_data)
        
        # Train model
        history = self.lstm_predictor.train(
            train_loader, val_loader,
            epochs=50,              # Reduced for faster demo
            learning_rate=0.001,
            patience=10
        )
        
        # Store training history
        self.lstm_history = history
        
        print("✅ LSTM training completed!")
        
    def evaluate_all_models(self):
        """Evaluate all models on test data."""
        print("\n📊 Evaluating all models...")
        
        self.results = {}
        
        # Evaluate LSTM
        print("\n🧠 Evaluating LSTM model...")
        lstm_metrics = self.lstm_predictor.evaluate(self.test_data)
        self.results['LSTM'] = lstm_metrics
        
        # Evaluate baseline models
        for model_name, model_info in self.baseline_models.items():
            print(f"\n🤖 Evaluating {model_name} model...")
            metrics = self.evaluate_baseline_model(model_info, self.test_data)
            self.results[model_name] = metrics
            
    def evaluate_baseline_model(self, model_info, test_data):
        """Evaluate a baseline model."""
        from sklearn.metrics import mean_squared_error, r2_score
        
        try:
            model = model_info['model']
            scaler = model_info.get('scaler')
            feature_cols = model_info.get('features')
            
            # If no features specified, use all numeric columns except target/identifiers
            if feature_cols is None:
                feature_cols = [col for col in test_data.columns 
                               if col not in ['LapTime', 'Driver', 'Team', 'DriverNumber'] and 
                               test_data[col].dtype in ['int64', 'float64', 'int32', 'float32']]
                model_info['features'] = feature_cols  # Store for future use
            
            # Prepare test features - only use columns that exist in test data
            available_features = [col for col in feature_cols if col in test_data.columns]
            X_test = test_data[available_features].fillna(0)
            y_test = test_data['LapTime']
            
            print(f"   Using {len(available_features)} features for {type(model).__name__}")
            
            # Scale if needed
            if scaler is not None:
                X_test = scaler.transform(X_test)
                
            # Make predictions
            predictions = model.predict(X_test)
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            r2 = r2_score(y_test, predictions)
            mae = np.mean(np.abs(y_test - predictions))
            
            metrics = {
                'rmse': rmse,
                'r2': r2,
                'mae': mae,
                'n_predictions': len(predictions)
            }
            
            print(f"   RMSE: {rmse:.4f} seconds")
            print(f"   R²: {r2:.4f}")
            print(f"   MAE: {mae:.4f} seconds")
            
            return metrics
            
        except Exception as e:
            print(f"   ❌ Error evaluating model: {e}")
            print(f"   Model type: {type(model_info.get('model', 'Unknown'))}")
            print(f"   Features: {feature_cols[:5] if feature_cols else 'None'}...")
            return {'rmse': float('inf'), 'r2': -1, 'mae': float('inf'), 'n_predictions': 0}
            
    def plot_comparison(self):
        """Plot model comparison results."""
        print("\n📈 Creating comparison plots...")
        
        # Extract metrics for plotting
        models = list(self.results.keys())
        rmse_values = [self.results[model]['rmse'] for model in models]
        r2_values = [self.results[model]['r2'] for model in models]
        mae_values = [self.results[model]['mae'] for model in models]
        
        # Create comparison plot
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # RMSE comparison
        bars1 = axes[0].bar(models, rmse_values, alpha=0.7, color=['red', 'blue', 'green', 'orange'])
        axes[0].set_title('RMSE Comparison (Lower is Better)')
        axes[0].set_ylabel('RMSE (seconds)')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars1, rmse_values):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
        
        # R² comparison
        bars2 = axes[1].bar(models, r2_values, alpha=0.7, color=['red', 'blue', 'green', 'orange'])
        axes[1].set_title('R² Comparison (Higher is Better)')
        axes[1].set_ylabel('R² Score')
        axes[1].tick_params(axis='x', rotation=45)
        
        for bar, value in zip(bars2, r2_values):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
        
        # MAE comparison
        bars3 = axes[2].bar(models, mae_values, alpha=0.7, color=['red', 'blue', 'green', 'orange'])
        axes[2].set_title('MAE Comparison (Lower is Better)')
        axes[2].set_ylabel('MAE (seconds)')
        axes[2].tick_params(axis='x', rotation=45)
        
        for bar, value in zip(bars3, mae_values):
            axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Plot LSTM training history if available
        if hasattr(self, 'lstm_history'):
            self.lstm_predictor.plot_training_history(self.lstm_history)
            
    def generate_report(self):
        """Generate a comprehensive comparison report."""
        print("\n📄 Generating comparison report...")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'data_info': {
                'total_samples': len(self.lap_data),
                'train_samples': len(self.train_data),
                'test_samples': len(self.test_data),
                'features_count': len([col for col in self.train_data.columns 
                                     if self.train_data[col].dtype in ['int64', 'float64']])
            },
            'model_results': self.results,
            'best_model': min(self.results.keys(), key=lambda k: self.results[k]['rmse']),
            'improvements': {}
        }
        
        # Calculate improvements over baseline
        baseline_rmse = min([self.results[k]['rmse'] for k in self.results.keys() if k != 'LSTM'])
        lstm_rmse = self.results['LSTM']['rmse']
        
        report['improvements']['rmse_improvement'] = (baseline_rmse - lstm_rmse) / baseline_rmse * 100
        report['improvements']['best_baseline_rmse'] = baseline_rmse
        report['improvements']['lstm_rmse'] = lstm_rmse
        
        # Save report
        with open('lstm_integration_report.json', 'w') as f:
            json.dump(report, f, indent=2)
            
        print(f"✅ Report saved to lstm_integration_report.json")
        
        # Print summary
        print(f"\n🏆 RESULTS SUMMARY:")
        print(f"   Best Model: {report['best_model']}")
        print(f"   LSTM RMSE: {lstm_rmse:.4f} seconds")
        print(f"   Best Baseline RMSE: {baseline_rmse:.4f} seconds")
        if lstm_rmse < baseline_rmse:
            improvement = (baseline_rmse - lstm_rmse) / baseline_rmse * 100
            print(f"   🎉 LSTM improved by {improvement:.1f}%!")
        else:
            degradation = (lstm_rmse - baseline_rmse) / baseline_rmse * 100
            print(f"   📉 LSTM performed {degradation:.1f}% worse than baseline")
            print(f"   💡 Try: longer training, more data, or different architecture")
            
    def run_full_comparison(self):
        """Run the complete model comparison pipeline."""
        print("🚀 Starting F1 LSTM Model Integration Test")
        print("=" * 60)
        
        # Step 1: Load data
        self.load_data()
        
        # Step 2: Load baseline models
        self.load_baseline_models()
        
        # Step 3: Train LSTM model
        self.train_lstm_model()
        
        # Step 4: Evaluate all models
        self.evaluate_all_models()
        
        # Step 5: Plot comparison
        self.plot_comparison()
        
        # Step 6: Generate report
        self.generate_report()
        
        print("\n✅ Integration test completed!")
        print("\n🎯 Next steps:")
        print("1. Review the comparison plots and report")
        print("2. If LSTM performs well, integrate into main pipeline")
        print("3. Experiment with different LSTM architectures")
        print("4. Consider ensemble methods combining LSTM + baseline models")


def main():
    """Main function to run the integration test."""
    # Initialize comparison
    comparison = F1ModelComparison()
    
    # Run full comparison
    comparison.run_full_comparison()


if __name__ == "__main__":
    main()
