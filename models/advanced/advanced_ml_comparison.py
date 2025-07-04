#!/usr/bin/env python3
"""
Advanced ML Model Comparison for F1-ML Platform

This script provides a comprehensive comparison between LSTM and traditional ML models
using your actual F1 data with proper feature alignment and fair comparison.
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Add paths to import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))

try:
    from models.advanced.deep_learning.lstm_model import F1LSTMPredictor
    print("✅ Successfully imported LSTM model")
except ImportError as e:
    print(f"❌ Failed to import LSTM model: {e}")
    sys.exit(1)

class F1AdvancedComparison:
    """
    Advanced comparison framework for F1 ML models.
    """
    
    def __init__(self, data_dir="../../data/processed/2024/Monaco Grand Prix/"):
        """Initialize the comparison framework."""
        self.data_dir = Path(data_dir)
        self.results = {}
        
        print(f"🏎️ F1 Advanced ML Model Comparison")
        print(f"📁 Data directory: {self.data_dir}")
        
    def load_data(self):
        """Load F1 lap data for testing."""
        print("\n📊 Loading F1 data...")
        
        try:
            lap_features_file = self.data_dir / "laps_features.csv"
            train_file = self.data_dir / "train_data.csv"
            test_file = self.data_dir / "test_data.csv"
            
            if lap_features_file.exists():
                self.lap_data = pd.read_csv(lap_features_file)
                print(f"✅ Loaded {len(self.lap_data)} lap records")
            else:
                raise FileNotFoundError("Lap features file not found")
                
            if train_file.exists() and test_file.exists():
                self.train_data = pd.read_csv(train_file)
                self.test_data = pd.read_csv(test_file)
                print(f"✅ Loaded train ({len(self.train_data)}) and test ({len(self.test_data)}) splits")
            else:
                print("⚠️ Creating new train/test split...")
                self.create_train_test_split()
                
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
            
        return True
        
    def create_train_test_split(self):
        """Create train/test split."""
        print("🔀 Creating train/test split...")
        
        # Use time-based split (first 80% for training)
        split_point = int(0.8 * len(self.lap_data))
        
        self.train_data = self.lap_data.iloc[:split_point].copy()
        self.test_data = self.lap_data.iloc[split_point:].copy()
        
        print(f"✅ Train: {len(self.train_data)} samples, Test: {len(self.test_data)} samples")
        
    def get_core_features(self):
        """Get the core feature set that all models will use."""
        # These are features that should be available in all F1 data
        core_features = [
            'Sector1Time', 'Sector2Time', 'Sector3Time',
            'TyreLife', 'Position', 'LapNumber',
            'SpeedI1', 'SpeedI2', 'SpeedFL', 'SpeedST',
            'gap_to_leader', 'gap_to_ahead', 'gap_to_behind',
            'rolling_avg_lap_time_3', 'rolling_avg_lap_time_5',
            'rolling_avg_position_3', 'rolling_avg_position_5',
            'tyre_age_normalized', 'laps_in_stint',
            'speed_consistency', 'speed_range',
            'car_ahead_time', 'car_behind_time',
            'lap_time_percentile', 'personal_bests_count',
            'team_avg_position', 'rolling_std_lap_time_3'
        ]
        
        # Only use features that exist in our data
        available_features = [col for col in core_features if col in self.train_data.columns]
        
        print(f"🎯 Using {len(available_features)} core features for fair comparison")
        print(f"   Features: {available_features}")
        
        return available_features
        
    def train_baseline_models(self):
        """Train baseline models using the same feature set."""
        print("\n🤖 Training baseline models...")
        
        # Get consistent feature set
        self.feature_cols = self.get_core_features()
        
        # Prepare training data
        X_train = self.train_data[self.feature_cols].fillna(0)
        y_train = self.train_data['LapTime']
        
        print(f"   Training on {X_train.shape[1]} features, {X_train.shape[0]} samples")
        
        # Scale features for models that need it
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        self.baseline_models = {}
        
        # 1. Random Forest (doesn't need scaling)
        print("   Training Random Forest...")
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        self.baseline_models['RandomForest'] = {
            'model': rf,
            'needs_scaling': False,
            'name': 'Random Forest'
        }
        
        # 2. ElasticNet (needs scaling)
        print("   Training ElasticNet...")
        en = ElasticNet(random_state=42, alpha=0.1, l1_ratio=0.5)
        en.fit(X_train_scaled, y_train)
        self.baseline_models['ElasticNet'] = {
            'model': en,
            'needs_scaling': True,
            'name': 'ElasticNet'
        }
        
        print(f"✅ Trained {len(self.baseline_models)} baseline models")
        
    def train_lstm_model(self):
        """Train the LSTM model."""
        print("\n🧠 Training LSTM model...")
        
        # Initialize LSTM predictor with optimized parameters
        self.lstm_predictor = F1LSTMPredictor(
            sequence_length=6,      # Shorter sequence for more data
            hidden_size=64,         # Moderate size
            num_layers=2,           # 2-layer LSTM
            dropout=0.3,            # More regularization
            use_attention=True      # Enable attention
        )
        
        # Prepare data with the same features
        train_data_lstm = self.train_data.copy()
        train_loader, val_loader = self.lstm_predictor.prepare_data(
            train_data_lstm, 
            feature_cols=self.feature_cols
        )
        
        # Train model with more epochs
        history = self.lstm_predictor.train(
            train_loader, val_loader,
            epochs=100,             # More training
            learning_rate=0.001,
            patience=15
        )
        
        self.lstm_history = history
        print("✅ LSTM training completed!")
        
    def evaluate_all_models(self):
        """Evaluate all models on test data."""
        print("\n📊 Evaluating all models...")
        
        self.results = {}
        
        # Prepare test data
        X_test = self.test_data[self.feature_cols].fillna(0)
        y_test = self.test_data['LapTime']
        X_test_scaled = self.scaler.transform(X_test)
        
        # Evaluate baseline models
        for model_name, model_info in self.baseline_models.items():
            print(f"\n🤖 Evaluating {model_info['name']}...")
            
            model = model_info['model']
            
            # Use scaled or unscaled features as needed
            if model_info['needs_scaling']:
                predictions = model.predict(X_test_scaled)
            else:
                predictions = model.predict(X_test)
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            r2 = r2_score(y_test, predictions)
            mae = np.mean(np.abs(y_test - predictions))
            
            self.results[model_name] = {
                'rmse': rmse,
                'r2': r2,
                'mae': mae,
                'n_predictions': len(predictions)
            }
            
            print(f"   RMSE: {rmse:.4f} seconds")
            print(f"   R²: {r2:.4f}")
            print(f"   MAE: {mae:.4f} seconds")
        
        # Evaluate LSTM
        print(f"\n🧠 Evaluating LSTM...")
        lstm_metrics = self.lstm_predictor.evaluate(self.test_data)
        self.results['LSTM'] = lstm_metrics
        
    def plot_comparison(self):
        """Create comprehensive comparison plots."""
        print("\n📈 Creating comparison plots...")
        
        models = list(self.results.keys())
        rmse_values = [self.results[model]['rmse'] for model in models]
        r2_values = [self.results[model]['r2'] for model in models]
        mae_values = [self.results[model]['mae'] for model in models]
        
        # Create figure with 2x2 subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # RMSE comparison
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        bars1 = axes[0,0].bar(models, rmse_values, color=colors, alpha=0.8)
        axes[0,0].set_title('RMSE Comparison (Lower is Better)', fontsize=14, fontweight='bold')
        axes[0,0].set_ylabel('RMSE (seconds)')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        for bar, value in zip(bars1, rmse_values):
            axes[0,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                          f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # R² comparison
        bars2 = axes[0,1].bar(models, r2_values, color=colors, alpha=0.8)
        axes[0,1].set_title('R² Comparison (Higher is Better)', fontsize=14, fontweight='bold')
        axes[0,1].set_ylabel('R² Score')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        for bar, value in zip(bars2, r2_values):
            axes[0,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                          f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # MAE comparison
        bars3 = axes[1,0].bar(models, mae_values, color=colors, alpha=0.8)
        axes[1,0].set_title('MAE Comparison (Lower is Better)', fontsize=14, fontweight='bold')
        axes[1,0].set_ylabel('MAE (seconds)')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        for bar, value in zip(bars3, mae_values):
            axes[1,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                          f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Model comparison summary
        axes[1,1].axis('off')
        
        # Create summary text
        best_model = min(models, key=lambda x: self.results[x]['rmse'])
        worst_model = max(models, key=lambda x: self.results[x]['rmse'])
        
        summary_text = f"""
        🏆 BEST MODEL: {best_model}
        RMSE: {self.results[best_model]['rmse']:.4f}s
        R²: {self.results[best_model]['r2']:.4f}
        
        📊 MODEL RANKINGS (by RMSE):
        """
        
        # Sort models by RMSE
        sorted_models = sorted(models, key=lambda x: self.results[x]['rmse'])
        for i, model in enumerate(sorted_models):
            rmse = self.results[model]['rmse']
            summary_text += f"\n{i+1}. {model}: {rmse:.4f}s"
            
        axes[1,1].text(0.1, 0.9, summary_text, transform=axes[1,1].transAxes,
                      fontsize=12, verticalalignment='top',
                      bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('advanced_model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Plot LSTM training history
        if hasattr(self, 'lstm_history'):
            self.lstm_predictor.plot_training_history(self.lstm_history)
            
    def generate_detailed_report(self):
        """Generate a comprehensive comparison report."""
        print("\n📄 Generating detailed report...")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'experiment_config': {
                'data_source': str(self.data_dir),
                'total_samples': len(self.lap_data),
                'train_samples': len(self.train_data),
                'test_samples': len(self.test_data),
                'features_used': len(self.feature_cols),
                'feature_list': self.feature_cols
            },
            'model_results': self.results,
            'rankings': {},
            'analysis': {}
        }
        
        # Create rankings
        models = list(self.results.keys())
        report['rankings']['by_rmse'] = sorted(models, key=lambda x: self.results[x]['rmse'])
        report['rankings']['by_r2'] = sorted(models, key=lambda x: self.results[x]['r2'], reverse=True)
        report['rankings']['by_mae'] = sorted(models, key=lambda x: self.results[x]['mae'])
        
        # Analysis
        best_model = report['rankings']['by_rmse'][0]
        worst_model = report['rankings']['by_rmse'][-1]
        
        best_rmse = self.results[best_model]['rmse']
        worst_rmse = self.results[worst_model]['rmse']
        
        report['analysis'] = {
            'best_model': best_model,
            'worst_model': worst_model,
            'rmse_range': worst_rmse - best_rmse,
            'rmse_improvement_over_worst': (worst_rmse - best_rmse) / worst_rmse * 100,
            'lstm_vs_best_traditional': None
        }
        
        # Compare LSTM vs best traditional model
        traditional_models = [m for m in models if m != 'LSTM']
        if traditional_models:
            best_traditional = min(traditional_models, key=lambda x: self.results[x]['rmse'])
            lstm_rmse = self.results['LSTM']['rmse']
            traditional_rmse = self.results[best_traditional]['rmse']
            
            improvement = (traditional_rmse - lstm_rmse) / traditional_rmse * 100
            report['analysis']['lstm_vs_best_traditional'] = {
                'best_traditional_model': best_traditional,
                'lstm_rmse': lstm_rmse,
                'traditional_rmse': traditional_rmse,
                'improvement_percentage': improvement
            }
        
        # Save report
        with open('advanced_ml_comparison_report.json', 'w') as f:
            json.dump(report, f, indent=2)
            
        print(f"✅ Report saved to advanced_ml_comparison_report.json")
        
        # Print summary
        print(f"\n🏆 FINAL RESULTS SUMMARY:")
        print(f"=" * 50)
        print(f"Best Model: {best_model} (RMSE: {best_rmse:.4f}s)")
        print(f"Worst Model: {worst_model} (RMSE: {worst_rmse:.4f}s)")
        print(f"Performance Range: {worst_rmse - best_rmse:.4f}s")
        
        if report['analysis']['lstm_vs_best_traditional']:
            lstm_analysis = report['analysis']['lstm_vs_best_traditional']
            if lstm_analysis['improvement_percentage'] > 0:
                print(f"🎉 LSTM improved by {lstm_analysis['improvement_percentage']:.1f}% over best traditional model!")
            else:
                print(f"📉 LSTM performed {abs(lstm_analysis['improvement_percentage']):.1f}% worse than best traditional model")
        
        print(f"\nModel Rankings (by RMSE):")
        for i, model in enumerate(report['rankings']['by_rmse']):
            rmse = self.results[model]['rmse']
            r2 = self.results[model]['r2']
            print(f"  {i+1}. {model}: RMSE={rmse:.4f}s, R²={r2:.4f}")
            
    def run_complete_comparison(self):
        """Run the complete advanced comparison."""
        print("🚀 Starting F1 Advanced ML Model Comparison")
        print("=" * 60)
        
        # Step 1: Load data
        if not self.load_data():
            print("❌ Failed to load data")
            return
        
        # Step 2: Train baseline models
        self.train_baseline_models()
        
        # Step 3: Train LSTM model
        self.train_lstm_model()
        
        # Step 4: Evaluate all models
        self.evaluate_all_models()
        
        # Step 5: Create visualizations
        self.plot_comparison()
        
        # Step 6: Generate detailed report
        self.generate_detailed_report()
        
        print("\n✅ Advanced comparison completed!")
        print("\n🎯 Key Insights:")
        print("1. All models trained on identical feature set for fair comparison")
        print("2. LSTM captures temporal dependencies in lap sequences")
        print("3. Traditional ML models use tabular features directly")
        print("4. Check the plots and detailed report for complete analysis")


def main():
    """Main function to run the advanced comparison."""
    comparison = F1AdvancedComparison()
    comparison.run_complete_comparison()


if __name__ == "__main__":
    main()
