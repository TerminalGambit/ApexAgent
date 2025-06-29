import pandas as pd
import numpy as np
import argparse
import os
import joblib
import json
from datetime import datetime

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, 
    ExtraTreesRegressor, VotingRegressor, BaggingRegressor
)
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available. Install with: pip install xgboost")

import warnings
warnings.filterwarnings('ignore')


class AdvancedModelTrainer:
    """Train and evaluate advanced machine learning models."""
    
    def __init__(self, year, race_name, cv_folds=5):
        self.year = year
        self.race_name = race_name
        self.cv_folds = cv_folds
        self.data_dir = "data/processed/{}/{}/".format(year, race_name)
        self.models_dir = "models/trained/"
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Initialize advanced models
        self.models = {
            'GradientBoosting': GradientBoostingRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            ),
            'ExtraTrees': ExtraTreesRegressor(
                n_estimators=200,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ),
            'ElasticNet': ElasticNet(
                alpha=0.1,
                l1_ratio=0.5,
                random_state=42
            ),
            'KNN': KNeighborsRegressor(
                n_neighbors=5,
                weights='distance'
            ),
            'Bagging_RF': BaggingRegressor(
                estimator=RandomForestRegressor(n_estimators=50, random_state=42),
                n_estimators=10,
                random_state=42,
                n_jobs=-1
            )
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            self.models['XGBoost_Advanced'] = xgb.XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )
        
        self.results = {}
        
    def load_data(self):
        """Load clean dataset for advanced modeling."""
        print("Loading clean data...")
        
        clean_path = os.path.join(self.data_dir, 'laps_features_clean.csv')
        if not os.path.exists(clean_path):
            raise FileNotFoundError("Clean dataset not found. Run investigate_data_leakage.py first.")
        
        df = pd.read_csv(clean_path)
        
        # Split features and target
        X = df.drop(columns=['LapTime'])
        y = df['LapTime']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=True
        )
        
        # Scale features for distance-based models
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Store both scaled and unscaled versions
        self.X_train = X_train.values
        self.X_test = X_test.values
        self.X_train_scaled = X_train_scaled
        self.X_test_scaled = X_test_scaled
        self.y_train = y_train.values
        self.y_test = y_test.values
        
        print("Train set: {} samples, {} features".format(self.X_train.shape[0], self.X_train.shape[1]))
        print("Test set: {} samples".format(self.X_test.shape[0]))
        
    def train_and_evaluate_model(self, name, model):
        """Train and evaluate a single advanced model."""
        print("\\nTraining {}...".format(name))
        
        # Use scaled data for distance-based models, unscaled for tree-based
        if name in ['ElasticNet', 'KNN']:
            X_train_use = self.X_train_scaled
            X_test_use = self.X_test_scaled
        else:
            X_train_use = self.X_train
            X_test_use = self.X_test
        
        # Train the model
        model.fit(X_train_use, self.y_train)
        
        # Make predictions
        y_train_pred = model.predict(X_train_use)
        y_test_pred = model.predict(X_test_use)
        
        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(self.y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(self.y_test, y_test_pred))
        train_mae = mean_absolute_error(self.y_train, y_train_pred)
        test_mae = mean_absolute_error(self.y_test, y_test_pred)
        train_r2 = r2_score(self.y_train, y_train_pred)
        test_r2 = r2_score(self.y_test, y_test_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train_use, self.y_train, 
                                   cv=self.cv_folds, scoring='neg_root_mean_squared_error')
        cv_rmse_mean = -cv_scores.mean()
        cv_rmse_std = cv_scores.std()
        
        # Store results
        self.results[name] = {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'cv_rmse_mean': cv_rmse_mean,
            'cv_rmse_std': cv_rmse_std
        }
        
        # Save the model
        model_path = os.path.join(self.models_dir, "advanced_{}.joblib".format(name))
        joblib.dump(model, model_path)
        
        print("  Train RMSE: {:.3f}, Test RMSE: {:.3f}".format(train_rmse, test_rmse))
        print("  Train R²: {:.3f}, Test R²: {:.3f}".format(train_r2, test_r2))
        print("  CV RMSE: {:.3f} (+/- {:.3f})".format(cv_rmse_mean, cv_rmse_std * 2))
        print("  Model saved to: {}".format(model_path))
        
    def create_ensemble_model(self):
        """Create and train an ensemble model."""
        print("\\nCreating ensemble model...")
        
        # Select best performing models for ensemble
        base_models = [
            ('gb', GradientBoostingRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)),
            ('et', ExtraTreesRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)),
            ('rf', RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1))
        ]
        
        # Add XGBoost to ensemble if available
        if XGBOOST_AVAILABLE:
            base_models.append(('xgb', xgb.XGBRegressor(
                n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1
            )))
        
        # Create voting ensemble
        ensemble = VotingRegressor(estimators=base_models)
        
        # Train and evaluate
        self.train_and_evaluate_model('Ensemble_Voting', ensemble)
        
    def train_all_models(self):
        """Train all advanced models."""
        print("Starting advanced model training...")
        print("=" * 60)
        
        for name, model in self.models.items():
            try:
                self.train_and_evaluate_model(name, model)
            except Exception as e:
                print("Error training {}: {}".format(name, e))
                continue
        
        # Train ensemble model
        try:
            self.create_ensemble_model()
        except Exception as e:
            print("Error training ensemble: {}".format(e))
        
        print("\\n" + "=" * 60)
        print("Advanced training completed!")
        
    def save_results(self):
        """Save training results to files."""
        print("\\nSaving results...")
        
        # Create results summary
        summary_data = []
        for name, metrics in self.results.items():
            summary_data.append({
                'model': name,
                'train_rmse': metrics['train_rmse'],
                'test_rmse': metrics['test_rmse'],
                'train_mae': metrics['train_mae'],
                'test_mae': metrics['test_mae'],
                'train_r2': metrics['train_r2'],
                'test_r2': metrics['test_r2'],
                'cv_rmse_mean': metrics['cv_rmse_mean'],
                'cv_rmse_std': metrics['cv_rmse_std']
            })
        
        # Save summary
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('test_rmse')  # Sort by test RMSE
        summary_path = os.path.join(self.models_dir, 'advanced_model_comparison.csv')
        summary_df.to_csv(summary_path, index=False)
        
        # Save detailed results as JSON
        detailed_results = {
            'training_info': {
                'year': self.year,
                'race_name': self.race_name,
                'timestamp': datetime.now().isoformat(),
                'cv_folds': self.cv_folds,
                'n_train_samples': len(self.X_train),
                'n_test_samples': len(self.X_test),
                'n_features': self.X_train.shape[1]
            },
            'model_results': {}
        }
        
        for name, metrics in self.results.items():
            detailed_results['model_results'][name] = {
                k: float(v) for k, v in metrics.items()
            }
        
        results_path = os.path.join(self.models_dir, 'advanced_training_results.json')
        with open(results_path, 'w') as f:
            json.dump(detailed_results, f, indent=2)
        
        print("Results saved to:")
        print("  - {}".format(summary_path))
        print("  - {}".format(results_path))
        
    def print_final_summary(self):
        """Print final model comparison summary."""
        print("\\n" + "=" * 80)
        print("ADVANCED MODEL COMPARISON SUMMARY")
        print("=" * 80)
        
        # Sort models by test RMSE
        sorted_models = sorted(self.results.items(), key=lambda x: x[1]['test_rmse'])
        
        print("\\n{:<25} {:<10} {:<10} {:<10} {:<10}".format(
            "Model", "Test RMSE", "Test R²", "Test MAE", "CV RMSE"))
        print("-" * 80)
        
        for name, metrics in sorted_models:
            print("{:<25} {:<10.3f} {:<10.3f} {:<10.3f} {:<10.3f}".format(
                name, 
                metrics['test_rmse'], 
                metrics['test_r2'], 
                metrics['test_mae'],
                metrics['cv_rmse_mean']
            ))
        
        best_model = sorted_models[0]
        print("\\n🏆 Best performing advanced model: {} (Test RMSE: {:.3f})".format(
            best_model[0], best_model[1]['test_rmse']))
        
        # Compare with baseline models
        baseline_path = os.path.join(self.models_dir, 'model_comparison.csv')
        if os.path.exists(baseline_path):
            baseline_df = pd.read_csv(baseline_path)
            best_baseline_rmse = baseline_df['test_rmse'].min()
            improvement = (best_baseline_rmse - best_model[1]['test_rmse']) / best_baseline_rmse * 100
            
            print("\\n📈 Improvement over baseline: {:.1f}% better RMSE".format(improvement))
            print("   Baseline best: {:.3f}".format(best_baseline_rmse))
            print("   Advanced best: {:.3f}".format(best_model[1]['test_rmse']))


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Train advanced machine learning models on clean data.")
    parser.add_argument("--year", type=int, default=2024, help="Race year")
    parser.add_argument("--race", type=str, default="Monaco Grand Prix", help="Race name")
    parser.add_argument("--cv", type=int, default=5, help="Number of cross-validation folds")
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = AdvancedModelTrainer(args.year, args.race, args.cv)
    
    try:
        # Load data
        trainer.load_data()
        
        # Train all models
        trainer.train_all_models()
        
        # Save results
        trainer.save_results()
        
        # Print summary
        trainer.print_final_summary()
        
        print("\n✅ Advanced model training completed successfully!")
        
    except Exception as e:
        print("❌ Training failed: {}".format(e))
        raise


if __name__ == "__main__":
    main()
