import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import joblib
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_absolute_error
from .api_integration import F1APIIntegration
from .season_analysis import SeasonStandingsAnalyzer


class RacePredictionEngine:
    """
    Iterative F1 race prediction system that updates predictions as new data becomes available
    """
    
    def __init__(self, year=2025, track_name="Austria GP"):
        self.year = year
        self.track_name = track_name
        self.output_dir = f"data/predictions/{year}/"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize F1 API integration for real data
        self.api_year = 2024 if year == 2025 else year  # Use 2024 data for 2025 predictions
        self.api_integration = F1APIIntegration(year=self.api_year)
        
        # Load real F1 data
        self.season_data = self._load_real_season_data()
        
        # Initialize prediction stages
        self.prediction_stages = {
            'baseline': None,      # Pre-practice prediction based on season form
            'practice': None,      # After practice sessions
            'qualifying': None,    # After qualifying
            'pre_race': None,      # Final prediction before race start
            'actual': None         # Actual race results (for comparison)
        }
        
        # Track prediction accuracy over stages
        self.accuracy_tracking = {
            'stage': [],
            'mae_position': [],
            'top3_accuracy': [],
            'winner_correct': [],
            'timestamp': []
        }
        
        # Driver and team mappings from real data
        self.driver_mapping = {}
        self.team_mapping = {}
        self._load_real_mappings()
        
        print(f"🏁 Race Prediction Engine initialized for {track_name} {year}")
        print(f"📊 Using real F1 data from {self.api_year} season")
    
    def _load_real_season_data(self):
        """Load real F1 season data from API"""
        print(f"🌐 Loading real F1 data for {self.api_year} season...")
        
        try:
            # Get comprehensive season data from API
            season_data = self.api_integration.get_comprehensive_season_data()
            
            if season_data and season_data.get('standings'):
                print(f"✅ Successfully loaded real F1 data")
                print(f"📊 Data sources: {', '.join(season_data.get('data_sources', []))}")
                return season_data
            else:
                print("⚠️ No real data available, will use fallback methods")
                return None
                
        except Exception as e:
            print(f"❌ Failed to load real F1 data: {e}")
            return None
    
    def _load_real_mappings(self):
        """Load driver and team mappings from real F1 data"""
        if not self.season_data or not self.season_data.get('standings'):
            # Create fallback mappings
            self._create_fallback_mappings()
            return
        
        standings = self.season_data['standings']
        
        # Load driver mappings from real standings
        if 'drivers' in standings:
            for i, driver_data in enumerate(standings['drivers']):
                # Use position-1 as driver_id for consistency
                driver_id = driver_data.get('position', i+1) - 1
                driver_name = driver_data.get('driver_name', f'Driver_{i}')
                
                # Extract first name or short name
                if ' ' in driver_name:
                    short_name = driver_name.split()[-1]  # Last name
                else:
                    short_name = driver_name
                
                self.driver_mapping[driver_id] = short_name
        
        # Load team mappings from real standings
        if 'constructors' in standings:
            for i, team_data in enumerate(standings['constructors']):
                team_id = i
                team_name = team_data.get('team_name', f'Team_{i}')
                # Simplify team names
                simplified_name = team_name.replace(' Honda RBPT', '').replace(' Mercedes', '').replace(' Aramco', '')
                self.team_mapping[team_id] = simplified_name
        
        print(f"📋 Loaded {len(self.driver_mapping)} drivers and {len(self.team_mapping)} teams from real data")
    
    def get_race_data(self):
        """Extract specific race data for the track from real F1 data"""
        if not self.season_data:
            return None
        
        # Try to get race results from API data
        race_results = self.season_data.get('race_results', {})
        
        # Look for the specific track in race results
        for round_num, race_data in race_results.items():
            race_name = race_data.get('race_name', '')
            # Match track names (handle variations)
            if (self.track_name.lower() in race_name.lower() or 
                race_name.lower() in self.track_name.lower()):
                return race_data
        
        # If no specific race found, return None (will simulate data)
        return None
    
    def stage_1_baseline_prediction(self):
        """Stage 1: Baseline prediction based on real season form and performance"""
        print("📊 Stage 1: Creating baseline prediction from real season data...")
        
        # Get current season standings from real data
        season_performance = {}
        
        if self.season_data and self.season_data.get('standings', {}).get('drivers'):
            drivers_standings = self.season_data['standings']['drivers']
            
            for i, driver_data in enumerate(drivers_standings):
                driver_id = i  # Use index as driver_id
                points = driver_data.get('points', 0)
                wins = driver_data.get('wins', 0)
                position = driver_data.get('position', i+1)
                
                # Calculate form score based on real performance
                max_points = drivers_standings[0].get('points', 1) if drivers_standings else 1
                form_score = points / max_points if max_points > 0 else 0
                
                # Calculate experience score based on wins and position
                experience_score = min(1.0, (wins * 0.1) + (1.0 - (position-1) / len(drivers_standings)))
                
                # Performance factor (inverse of position, normalized)
                performance_factor = 1.0 + (len(drivers_standings) - position) / len(drivers_standings) * 0.2
                
                season_performance[driver_id] = {
                    'name': self.driver_mapping.get(driver_id, driver_data.get('driver_name', f'Driver_{i}')),
                    'team': driver_data.get('team_name', 'Unknown'),
                    'form': form_score,
                    'experience': experience_score,
                    'performance': performance_factor,
                    'points': points,
                    'wins': wins,
                    'championship_position': position
                }
        else:
            # Fallback to dummy data if no real data available
            print("⚠️ No real standings data, using fallback performance data")
            season_performance = self._create_fallback_performance_data()
        
        # Create baseline prediction based on season form
        baseline_prediction = []
        for driver_id in sorted(season_performance.keys()):
            driver = season_performance[driver_id]
            
            # Calculate baseline score (lower is better for race position)
            baseline_score = (
                driver['performance'] * 0.5 +  # Current pace
                (1 - driver['form']) * 0.3 +   # Recent form (inverted)
                (1 - driver['experience']) * 0.2  # Experience (inverted)
            )
            
            baseline_prediction.append({
                'driver_id': driver_id,
                'driver_name': driver['name'],
                'team': driver['team'],
                'predicted_position': 0,  # Will be set after sorting
                'confidence': 0.4,  # Low confidence for baseline
                'baseline_score': baseline_score
            })
        
        # Sort by baseline score and assign positions
        baseline_prediction.sort(key=lambda x: x['baseline_score'])
        for i, pred in enumerate(baseline_prediction):
            pred['predicted_position'] = i + 1
            del pred['baseline_score']  # Remove intermediate score
        
        self.prediction_stages['baseline'] = {
            'predictions': baseline_prediction,
            'stage': 'baseline',
            'timestamp': datetime.now().isoformat(),
            'data_sources': ['season_form', 'driver_experience']
        }
        
        print(f"✅ Baseline prediction complete. Predicted winner: {baseline_prediction[0]['driver_name']}")
        return baseline_prediction
    
    def stage_2_practice_update(self, practice_data=None):
        """Stage 2: Update prediction with practice session data"""
        print("🏃 Stage 2: Updating prediction with practice data...")
        
        if practice_data is None:
            # Simulate practice data based on baseline + some variation
            practice_data = self._simulate_practice_data()
        
        # Update predictions based on practice performance
        updated_predictions = []
        baseline_preds = {p['driver_id']: p for p in self.prediction_stages['baseline']['predictions']}
        
        for driver_id, practice_info in practice_data.items():
            baseline = baseline_preds.get(driver_id, {})
            
            # Combine baseline with practice performance
            practice_factor = practice_info.get('best_time_rank', 10) / 20.0  # Normalize to 0-1
            baseline_factor = (baseline.get('predicted_position', 10) - 1) / 19.0  # Normalize to 0-1
            
            # Weighted combination (practice gets more weight)
            combined_score = practice_factor * 0.6 + baseline_factor * 0.4
            
            updated_predictions.append({
                'driver_id': driver_id,
                'driver_name': baseline.get('driver_name', f'Driver_{driver_id}'),
                'team': baseline.get('team', 'Unknown'),
                'predicted_position': 0,  # Will be set after sorting
                'confidence': 0.6,  # Higher confidence with practice data
                'practice_time': practice_info.get('best_time', 0),
                'combined_score': combined_score
            })
        
        # Sort and assign positions
        updated_predictions.sort(key=lambda x: x['combined_score'])
        for i, pred in enumerate(updated_predictions):
            pred['predicted_position'] = i + 1
            del pred['combined_score']  # Remove intermediate score
        
        self.prediction_stages['practice'] = {
            'predictions': updated_predictions,
            'stage': 'practice',
            'timestamp': datetime.now().isoformat(),
            'data_sources': ['season_form', 'practice_sessions']
        }
        
        print(f"✅ Practice update complete. New predicted winner: {updated_predictions[0]['driver_name']}")
        return updated_predictions
    
    def stage_3_qualifying_update(self):
        """Stage 3: Update prediction with qualifying results"""
        print("🏁 Stage 3: Updating prediction with qualifying data...")
        
        race_data = self.get_race_data()
        if not race_data or 'qualifying' not in race_data:
            print("⚠️ No qualifying data found, using practice predictions")
            return self.prediction_stages['practice']['predictions']
        
        qualifying_results = race_data['qualifying']
        
        # Update predictions based on qualifying performance
        updated_predictions = []
        practice_preds = {p['driver_id']: p for p in self.prediction_stages['practice']['predictions']}
        
        for qual_result in qualifying_results:
            driver_id = qual_result['driver_id']
            practice_pred = practice_preds.get(driver_id, {})
            
            # Qualifying position is highly predictive of race result
            qual_position = qual_result['position']
            practice_position = practice_pred.get('predicted_position', qual_position)
            
            # Weighted combination (qualifying gets high weight)
            predicted_position = qual_position * 0.7 + practice_position * 0.3
            
            updated_predictions.append({
                'driver_id': driver_id,
                'driver_name': qual_result['driver_name'],
                'team': qual_result['team'],
                'predicted_position': predicted_position,
                'confidence': 0.8,  # High confidence with qualifying data
                'qualifying_position': qual_position,
                'qualifying_time': qual_result['time']
            })
        
        # Sort by predicted position
        updated_predictions.sort(key=lambda x: x['predicted_position'])
        for i, pred in enumerate(updated_predictions):
            pred['predicted_position'] = i + 1
        
        self.prediction_stages['qualifying'] = {
            'predictions': updated_predictions,
            'stage': 'qualifying',
            'timestamp': datetime.now().isoformat(),
            'data_sources': ['season_form', 'practice_sessions', 'qualifying_results']
        }
        
        print(f"✅ Qualifying update complete. Final predicted winner: {updated_predictions[0]['driver_name']}")
        return updated_predictions
    
    def stage_4_pre_race_final(self, weather_conditions=None, strategy_intel=None):
        """Stage 4: Final pre-race prediction with weather and strategy"""
        print("🌤️ Stage 4: Final pre-race prediction with conditions...")
        
        qualifying_preds = self.prediction_stages['qualifying']['predictions'].copy()
        
        # Apply weather and strategy adjustments
        for pred in qualifying_preds:
            # Minor adjustments based on weather (simulate for now)
            weather_factor = np.random.normal(1.0, 0.05)  # Small weather impact
            strategy_factor = np.random.normal(1.0, 0.1)   # Strategy impact
            
            # Adjust position slightly
            current_pos = pred['predicted_position']
            adjusted_pos = current_pos * weather_factor * strategy_factor
            pred['predicted_position'] = max(1, min(20, adjusted_pos))
            pred['confidence'] = 0.85  # Very high confidence
        
        # Re-sort and assign final positions
        qualifying_preds.sort(key=lambda x: x['predicted_position'])
        for i, pred in enumerate(qualifying_preds):
            pred['predicted_position'] = i + 1
        
        self.prediction_stages['pre_race'] = {
            'predictions': qualifying_preds,
            'stage': 'pre_race',
            'timestamp': datetime.now().isoformat(),
            'data_sources': ['season_form', 'practice_sessions', 'qualifying_results', 'weather_conditions']
        }
        
        print(f"✅ Final prediction complete. Race winner: {qualifying_preds[0]['driver_name']}")
        return qualifying_preds
    
    def load_actual_results(self):
        """Load actual race results for comparison"""
        race_data = self.get_race_data()
        if not race_data or 'results' not in race_data:
            return None
        
        actual_results = []
        for result in race_data['results']:
            actual_results.append({
                'driver_id': result['driver_id'],
                'driver_name': result['driver_name'],
                'team': result['team'],
                'actual_position': result['position'],
                'points': result['points'],
                'status': result['status']
            })
        
        self.prediction_stages['actual'] = {
            'results': actual_results,
            'stage': 'actual',
            'timestamp': datetime.now().isoformat()
        }
        
        return actual_results
    
    def evaluate_predictions(self):
        """Evaluate prediction accuracy across all stages"""
        print("📈 Evaluating prediction accuracy...")
        
        actual_results = self.load_actual_results()
        if not actual_results:
            print("⚠️ No actual results available for evaluation")
            return None
        
        # Create actual position mapping
        actual_positions = {r['driver_id']: r['actual_position'] for r in actual_results}
        
        # Evaluate each stage
        evaluation_results = {}
        
        for stage_name, stage_data in self.prediction_stages.items():
            if stage_name == 'actual' or not stage_data:
                continue
            
            predictions = stage_data['predictions']
            
            # Calculate metrics
            position_errors = []
            top3_correct = 0
            winner_correct = False
            
            for pred in predictions:
                driver_id = pred['driver_id']
                predicted_pos = pred['predicted_position']
                actual_pos = actual_positions.get(driver_id, 20)
                
                # Position error
                position_errors.append(abs(predicted_pos - actual_pos))
                
                # Top 3 accuracy
                if predicted_pos <= 3 and actual_pos <= 3:
                    top3_correct += 1
                
                # Winner accuracy
                if predicted_pos == 1 and actual_pos == 1:
                    winner_correct = True
            
            # Calculate summary metrics
            mae_position = np.mean(position_errors)
            top3_accuracy = top3_correct / 3.0  # 3 possible top-3 positions
            
            evaluation_results[stage_name] = {
                'mae_position': mae_position,
                'top3_accuracy': top3_accuracy,
                'winner_correct': winner_correct,
                'total_predictions': len(predictions)
            }
            
            # Track for dashboard
            self.accuracy_tracking['stage'].append(stage_name)
            self.accuracy_tracking['mae_position'].append(mae_position)
            self.accuracy_tracking['top3_accuracy'].append(top3_accuracy)
            self.accuracy_tracking['winner_correct'].append(winner_correct)
            self.accuracy_tracking['timestamp'].append(datetime.now().isoformat())
        
        return evaluation_results
    
    def _simulate_practice_data(self):
        """Simulate practice session data based on baseline predictions"""
        practice_data = {}
        baseline_preds = self.prediction_stages['baseline']['predictions']
        
        for pred in baseline_preds:
            driver_id = pred['driver_id']
            # Simulate practice times with some randomness
            base_time = 75.0 + (pred['predicted_position'] - 1) * 0.5
            practice_time = base_time + np.random.normal(0, 0.3)
            
            practice_data[driver_id] = {
                'best_time': practice_time,
                'best_time_rank': pred['predicted_position'] + np.random.randint(-2, 3)
            }
        
        return practice_data
    
    def _convert_form_to_score(self, form_str):
        """Convert form string to numeric score (0-1, higher is better)"""
        form_mapping = {
            'excellent': 0.95,
            'strong': 0.8,
            'motivated': 0.75,
            'improving': 0.7,
            'solid': 0.65,
            'steady': 0.6,
            'consistent': 0.6,
            'reliable': 0.6,
            'professional': 0.55,
            'promising': 0.5,
            'fast-learning': 0.5,
            'developing': 0.45,
            'learning': 0.4,
            'declining': 0.3,
            'struggling': 0.2,
            'aggressive': 0.55  # Can be good or bad
        }
        return form_mapping.get(form_str.lower(), 0.5)
    
    def _convert_experience_to_score(self, experience_str):
        """Convert experience string to numeric score (0-1, higher is better)"""
        experience_mapping = {
            'legendary': 1.0,
            'high': 0.8,
            'medium': 0.6,
            'rookie': 0.2
        }
        return experience_mapping.get(experience_str.lower(), 0.5)
    
    def _create_fallback_mappings(self):
        """Create fallback driver and team mappings when real data is unavailable"""
        print("⚠️ Creating fallback driver/team mappings")
        
        # 2024/2025 driver lineup (realistic)
        fallback_drivers = [
            'Verstappen', 'Norris', 'Leclerc', 'Piastri', 'Sainz',
            'Russell', 'Hamilton', 'Perez', 'Alonso', 'Hulkenberg',
            'Gasly', 'Ocon', 'Albon', 'Stroll', 'Tsunoda',
            'Magnussen', 'Bottas', 'Zhou', 'Lawson', 'Colapinto'
        ]
        
        fallback_teams = [
            'Red Bull Racing', 'McLaren', 'Ferrari', 'Mercedes', 'Aston Martin',
            'Alpine', 'Williams', 'Haas', 'Sauber', 'RB'
        ]
        
        for i, driver in enumerate(fallback_drivers):
            self.driver_mapping[i] = driver
        
        for i, team in enumerate(fallback_teams):
            self.team_mapping[i] = team
    
    def _create_fallback_performance_data(self):
        """Create fallback performance data based on realistic 2024/2025 expectations"""
        # Realistic performance based on 2024 season trends
        fallback_performance = {}
        
        drivers_data = [
            {'name': 'Verstappen', 'team': 'Red Bull Racing', 'form': 0.95, 'performance': 0.98},
            {'name': 'Norris', 'team': 'McLaren', 'form': 0.9, 'performance': 0.96},
            {'name': 'Leclerc', 'team': 'Ferrari', 'form': 0.85, 'performance': 0.95},
            {'name': 'Piastri', 'team': 'McLaren', 'form': 0.8, 'performance': 0.93},
            {'name': 'Sainz', 'team': 'Williams', 'form': 0.75, 'performance': 0.92},
            {'name': 'Russell', 'team': 'Mercedes', 'form': 0.7, 'performance': 0.90},
            {'name': 'Hamilton', 'team': 'Ferrari', 'form': 0.75, 'performance': 0.91},
            {'name': 'Perez', 'team': 'Red Bull Racing', 'form': 0.6, 'performance': 0.88},
            {'name': 'Alonso', 'team': 'Aston Martin', 'form': 0.65, 'performance': 0.89},
            {'name': 'Hulkenberg', 'team': 'Haas', 'form': 0.6, 'performance': 0.85}
        ]
        
        for i, driver_data in enumerate(drivers_data):
            fallback_performance[i] = {
                'name': driver_data['name'],
                'team': driver_data['team'],
                'form': driver_data['form'],
                'experience': 0.7,  # Average experience
                'performance': driver_data['performance'],
                'points': max(0, 400 - i * 40),  # Decreasing points
                'wins': max(0, 8 - i),  # Decreasing wins
                'championship_position': i + 1
            }
        
        return fallback_performance
    
    def generate_prediction_report(self):
        """Generate comprehensive prediction report"""
        report = {
            'race_info': {
                'year': self.year,
                'track': self.track_name,
                'prediction_date': datetime.now().isoformat()
            },
            'prediction_stages': self.prediction_stages,
            'accuracy_evaluation': self.evaluate_predictions(),
            'summary': self._generate_summary()
        }
        
        # Save report
        report_path = os.path.join(self.output_dir, f'{self.track_name.replace(" ", "_")}_prediction_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"📊 Prediction report saved to: {report_path}")
        return report
    
    def _generate_summary(self):
        """Generate prediction summary"""
        if not self.prediction_stages['pre_race']:
            return {}
        
        final_predictions = self.prediction_stages['pre_race']['predictions']
        
        return {
            'predicted_winner': final_predictions[0]['driver_name'],
            'predicted_podium': [p['driver_name'] for p in final_predictions[:3]],
            'biggest_surprise': self._find_biggest_surprise(),
            'confidence_level': np.mean([p['confidence'] for p in final_predictions])
        }
    
    def _find_biggest_surprise(self):
        """Identify the biggest surprise in predictions vs baseline"""
        if not self.prediction_stages['baseline'] or not self.prediction_stages['pre_race']:
            return None
        
        baseline = {p['driver_id']: p['predicted_position'] for p in self.prediction_stages['baseline']['predictions']}
        final = {p['driver_id']: p['predicted_position'] for p in self.prediction_stages['pre_race']['predictions']}
        
        biggest_change = 0
        surprise_driver = None
        
        for driver_id in baseline.keys():
            change = baseline[driver_id] - final.get(driver_id, baseline[driver_id])
            if abs(change) > biggest_change:
                biggest_change = abs(change)
                surprise_driver = self.driver_mapping.get(driver_id, f'Driver_{driver_id}')
        
        return {
            'driver': surprise_driver,
            'position_change': biggest_change
        }
    
    def get_dashboard_data(self):
        """Prepare data for dashboard visualization"""
        dashboard_data = {
            'race_info': {
                'track': self.track_name,
                'year': self.year,
                'last_updated': datetime.now().isoformat()
            },
            'prediction_timeline': [],
            'accuracy_metrics': self.accuracy_tracking,
            'current_predictions': None,
            'actual_results': None
        }
        
        # Prediction timeline
        for stage_name, stage_data in self.prediction_stages.items():
            if stage_data and stage_name != 'actual':
                dashboard_data['prediction_timeline'].append({
                    'stage': stage_name,
                    'predictions': stage_data['predictions'][:10],  # Top 10
                    'timestamp': stage_data['timestamp']
                })
        
        # Current predictions (latest stage)
        if self.prediction_stages['pre_race']:
            dashboard_data['current_predictions'] = self.prediction_stages['pre_race']['predictions']
        elif self.prediction_stages['qualifying']:
            dashboard_data['current_predictions'] = self.prediction_stages['qualifying']['predictions']
        
        # Actual results
        if self.prediction_stages['actual']:
            dashboard_data['actual_results'] = self.prediction_stages['actual']['results']
        
        return dashboard_data


def main():
    """Test the race prediction system for Austria 2025"""
    print("🏎️ F1 Race Prediction System - Austria 2025")
    print("=" * 50)
    
    # Initialize predictor
    predictor = RacePredictionEngine(year=2025, track_name="Austria GP")
    
    try:
        # Run all prediction stages
        print("\n🚀 Running iterative prediction pipeline...")
        
        # Stage 1: Baseline
        baseline = predictor.stage_1_baseline_prediction()
        
        # Stage 2: Practice
        practice = predictor.stage_2_practice_update()
        
        # Stage 3: Qualifying
        qualifying = predictor.stage_3_qualifying_update()
        
        # Stage 4: Pre-race final
        final = predictor.stage_4_pre_race_final()
        
        # Load actual results and evaluate
        actual = predictor.load_actual_results()
        evaluation = predictor.evaluate_predictions()
        
        # Generate report
        report = predictor.generate_prediction_report()
        
        print("\n" + "=" * 50)
        print("✅ Prediction Pipeline Complete!")
        print(f"🏆 Predicted Winner: {final[0]['driver_name']}")
        print(f"🥉 Predicted Podium: {', '.join([p['driver_name'] for p in final[:3]])}")
        
        if actual:
            actual_winner = actual[0]['driver_name']
            print(f"🎯 Actual Winner: {actual_winner}")
            print(f"✅ Prediction Correct: {'Yes' if final[0]['driver_name'] == actual_winner else 'No'}")
        
        if evaluation:
            print(f"\n📈 Final Prediction Accuracy:")
            final_eval = evaluation.get('pre_race', {})
            print(f"  • Position MAE: {final_eval.get('mae_position', 0):.2f}")
            print(f"  • Top 3 Accuracy: {final_eval.get('top3_accuracy', 0)*100:.1f}%")
            print(f"  • Winner Correct: {'Yes' if final_eval.get('winner_correct', False) else 'No'}")
        
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        raise


if __name__ == "__main__":
    main()
