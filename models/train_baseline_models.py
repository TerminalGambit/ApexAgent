import pandas as pd
import numpy as np
import argparse
import os
import joblib
import json
from datetime import datetime

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available. Install with: pip install xgboost")

import warnings
warnings.filterwarnings('ignore')


class BaselineModelTrainer:
    """Train and evaluate baseline machine learning models."""
    
    def __init__(self, year, race_name, cv_folds=5):
        self.year = year
        self.race_name = race_name
        self.cv_folds = cv_folds
        self.data_dir = "data/processed/{}/{}/".format(year, race_name)
        self.models_dir = "models/trained/"
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Initialize models
        self.models = {
            'Linear_Regression': LinearRegression(),
            'Ridge_Regression': Ridge(alpha=1.0),
            'Lasso_Regression': Lasso(alpha=1.0),
            'Random_Forest': RandomForestRegressor(
                n_estimators=100, 
                max_depth=10, 
                random_state=42,
                n_jobs=-1
            ),
            'SVR': SVR(kernel='rbf', C=1.0, gamma='scale')
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            self.models['XGBoost'] = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            )
        
        self.results = {}
        
    def load_data(self):
        """Load prepared training and test data."""
        print("Loading prepared data...")
        
        train_path = os.path.join(self.data_dir, 'train_data.csv')
        test_path = os.path.join(self.data_dir, 'test_data.csv')
        
        if not os.path.exists(train_path) or not os.path.exists(test_path):
            raise FileNotFoundError("Prepared data not found. Run prepare_model_data.py first.")
        
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        
        # Separate features and target
        target_col = 'LapTime'  # Assuming LapTime is the target
        
        self.X_train = train_data.drop(columns=[target_col])
        self.y_train = train_data[target_col]
        self.X_test = test_data.drop(columns=[target_col])
        self.y_test = test_data[target_col]
        
        print("Train set: {} samples, {} features".format(len(self.X_train), len(self.X_train.columns)))
        print("Test set: {} samples".format(len(self.X_test)))
        
    def train_and_evaluate_model(self, name, model):
        """Train and evaluate a single model."""
        print("\\nTraining {}...".format(name))
        
        # Train the model
        model.fit(self.X_train, self.y_train)
        
        # Make predictions
        y_train_pred = model.predict(self.X_train)
        y_test_pred = model.predict(self.X_test)
        
        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(self.y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(self.y_test, y_test_pred))
        train_mae = mean_absolute_error(self.y_train, y_train_pred)
        test_mae = mean_absolute_error(self.y_test, y_test_pred)
        train_r2 = r2_score(self.y_train, y_train_pred)
        test_r2 = r2_score(self.y_test, y_test_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(model, self.X_train, self.y_train, 
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
            'cv_rmse_std': cv_rmse_std,
            'predictions_train': y_train_pred,
            'predictions_test': y_test_pred
        }
        
        # Save the model
        model_path = os.path.join(self.models_dir, "{}.joblib".format(name))
        joblib.dump(model, model_path)
        
        print("  Train RMSE: {:.3f}, Test RMSE: {:.3f}".format(train_rmse, test_rmse))
        print("  Train R²: {:.3f}, Test R²: {:.3f}".format(train_r2, test_r2))
        print("  CV RMSE: {:.3f} (+/- {:.3f})".format(cv_rmse_mean, cv_rmse_std * 2))
        print("  Model saved to: {}".format(model_path))
        
    def train_all_models(self):
        """Train all baseline models."""
        print("Starting baseline model training...")
        print("=" * 50)
        
        for name, model in self.models.items():
            try:
                self.train_and_evaluate_model(name, model)
            except Exception as e:
                print("Error training {}: {}".format(name, e))
                continue
        
        print("\\n" + "=" * 50)
        print("Training completed!")
        
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
        summary_path = os.path.join(self.models_dir, 'model_comparison.csv')
        summary_df.to_csv(summary_path, index=False)
        
        # Save detailed results
        detailed_results = {
            'training_info': {
                'year': self.year,
                'race_name': self.race_name,
                'timestamp': datetime.now().isoformat(),
                'cv_folds': self.cv_folds,
                'n_train_samples': len(self.X_train),
                'n_test_samples': len(self.X_test),
                'n_features': len(self.X_train.columns)
            },
            'model_results': {}
        }
        
        for name, metrics in self.results.items():
            detailed_results['model_results'][name] = {
                'train_rmse': float(metrics['train_rmse']),
                'test_rmse': float(metrics['test_rmse']),
                'train_mae': float(metrics['train_mae']),
                'test_mae': float(metrics['test_mae']),
                'train_r2': float(metrics['train_r2']),
                'test_r2': float(metrics['test_r2']),
                'cv_rmse_mean': float(metrics['cv_rmse_mean']),
                'cv_rmse_std': float(metrics['cv_rmse_std'])
            }
        
        # Save as JSON
        results_path = os.path.join(self.models_dir, 'training_results.json')
        with open(results_path, 'w') as f:
            json.dump(detailed_results, f, indent=2)
        
        print("Results saved to:")
        print("  - {}".format(summary_path))
        print("  - {}".format(results_path))
        
    def print_final_summary(self):
        """Print final model comparison summary."""
        print("\\n" + "=" * 70)
        print("FINAL MODEL COMPARISON SUMMARY")
        print("=" * 70)
        
        # Sort models by test RMSE
        sorted_models = sorted(self.results.items(), key=lambda x: x[1]['test_rmse'])
        
        print("\\n{:<20} {:<10} {:<10} {:<10} {:<10}".format(
            "Model", "Test RMSE", "Test R²", "Test MAE", "CV RMSE"))
        print("-" * 70)
        
        for name, metrics in sorted_models:
            print("{:<20} {:<10.3f} {:<10.3f} {:<10.3f} {:<10.3f}".format(
                name, 
                metrics['test_rmse'], 
                metrics['test_r2'], 
                metrics['test_mae'],
                metrics['cv_rmse_mean']
            ))
        
        best_model = sorted_models[0]
        print("\\n🏆 Best performing model: {} (Test RMSE: {:.3f})".format(
            best_model[0], best_model[1]['test_rmse']))
        
        # Check if we meet success criteria
        best_rmse = best_model[1]['test_rmse']
        best_r2 = best_model[1]['test_r2']
        
        print("\\n📊 Success Criteria Check:")
        print("  Target: RMSE < 2.0 seconds")
        print("  Achieved: RMSE = {:.3f} seconds {}".format(
            best_rmse, "✅ PASS" if best_rmse < 2.0 else "❌ FAIL"))
        
        print("  Target: R² > 0.85")
        print("  Achieved: R² = {:.3f} {}".format(
            best_r2, "✅ PASS" if best_r2 > 0.85 else "❌ FAIL"))


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Train baseline machine learning models.")
    parser.add_argument("--year", type=int, default=2024, help="Race year")
    parser.add_argument("--race", type=str, default="Monaco Grand Prix", help="Race name")
    parser.add_argument("--cv", type=int, default=5, help="Number of cross-validation folds")
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = BaselineModelTrainer(args.year, args.race, args.cv)
    
    try:
        # Load data
        trainer.load_data()
        
        # Train all models
        trainer.train_all_models()
        
        # Save results
        trainer.save_results()
        
        # Print summary
        trainer.print_final_summary()
        
        print("\\n✅ Baseline model training completed successfully!")
        
    except Exception as e:
        print("❌ Training failed: {}".format(e))
        raise


if __name__ == "__main__":
    main()
