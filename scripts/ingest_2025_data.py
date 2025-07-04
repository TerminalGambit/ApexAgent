#!/usr/bin/env python3
"""
2025 F1 Data Ingestion Script

This script automates the complete data pipeline for 2025 F1 season:
1. Ingests raw lap data using FastF1
2. Cleans the data
3. Engineers features 
4. Prepares model-ready datasets

Usage:
    python scripts/ingest_2025_data.py --races "Bahrain Grand Prix,Saudi Arabian Grand Prix" --year 2025
    python scripts/ingest_2025_data.py --all-available  # Process all available 2025 races
"""

import os
import sys
import argparse
import logging
import subprocess
from pathlib import Path
from typing import List, Dict, Optional
import json
from datetime import datetime

# Add project root to Python path
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.append(str(project_root))

# Import our data pipeline modules
from data_pipeline.api_integration import F1APIIntegration
from data_pipeline.feature_engineering import F1FeatureEngineer
from data_pipeline.prepare_model_data import prepare_model_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/2025_data_ingestion.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class F1DataPipeline2025:
    """Complete data pipeline for 2025 F1 season data"""
    
    def __init__(self, year: int = 2025):
        self.year = year
        self.api = F1APIIntegration(year=year)
        self.engineer = F1FeatureEngineer()
        
        # Create necessary directories
        self.base_dir = project_root / "data"
        self.raw_dir = self.base_dir / "raw" / str(year)
        self.processed_dir = self.base_dir / "processed" / str(year)
        self.logs_dir = project_root / "logs"
        
        for directory in [self.raw_dir, self.processed_dir, self.logs_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def get_available_races(self) -> List[Dict]:
        """Get list of available races for the 2025 season"""
        logger.info(f"Fetching available races for {self.year}...")
        
        try:
            schedule = self.api.fetch_race_schedule()
            if schedule:
                logger.info(f"Found {len(schedule)} races in {self.year} schedule")
                return schedule
            else:
                logger.warning("No race schedule found, using fallback race list")
                # Fallback list of major 2025 races (based on typical F1 calendar)
                return [
                    {"race_name": "Bahrain Grand Prix", "round": 1},
                    {"race_name": "Saudi Arabian Grand Prix", "round": 2}, 
                    {"race_name": "Australian Grand Prix", "round": 3},
                    {"race_name": "Chinese Grand Prix", "round": 4},
                    {"race_name": "Miami Grand Prix", "round": 5},
                    {"race_name": "Emilia Romagna Grand Prix", "round": 6},
                    {"race_name": "Monaco Grand Prix", "round": 7},
                    {"race_name": "Spanish Grand Prix", "round": 8},
                    {"race_name": "Canadian Grand Prix", "round": 9},
                    {"race_name": "Austrian Grand Prix", "round": 10},
                ]
        except Exception as e:
            logger.error(f"Failed to fetch race schedule: {e}")
            return []
    
    def ingest_race_data(self, race_name: str) -> bool:
        """Ingest raw lap data for a specific race"""
        logger.info(f"Ingesting data for {race_name}...")
        
        try:
            # Create race directory
            race_dir = self.raw_dir / race_name.replace(" ", "_")
            race_dir.mkdir(exist_ok=True)
            
            # Fetch lap data using FastF1
            lap_data = self.api.fetch_lap_data(race_name, session_type='R')
            
            if lap_data is not None and not lap_data.empty:
                # Save raw lap data
                output_path = race_dir / "laps.csv"
                lap_data.to_csv(output_path, index=False)
                logger.info(f"✅ Saved {len(lap_data)} laps to {output_path}")
                
                # Also get session metadata
                session_data = self.api.fetch_session_data(race_name, session_type='R')
                if session_data:
                    metadata_path = race_dir / "session_metadata.json"
                    with open(metadata_path, 'w') as f:
                        json.dump(session_data, f, indent=2, default=str)
                    logger.info(f"✅ Saved session metadata to {metadata_path}")
                
                return True
            else:
                logger.warning(f"❌ No lap data available for {race_name}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to ingest {race_name}: {e}")
            return False
    
    def clean_race_data(self, race_name: str) -> bool:
        """Clean raw lap data for a specific race"""
        logger.info(f"Cleaning data for {race_name}...")
        
        try:
            # Use the existing clean_laps.py script
            race_dir_name = race_name.replace(" ", "_")
            cmd = [
                sys.executable,
                str(project_root / "data_pipeline" / "clean_laps.py"),
                "--year", str(self.year),
                "--race", race_name
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=project_root)
            
            if result.returncode == 0:
                logger.info(f"✅ Successfully cleaned data for {race_name}")
                return True
            else:
                logger.error(f"❌ Failed to clean {race_name}: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error cleaning {race_name}: {e}")
            return False
    
    def engineer_race_features(self, race_name: str) -> bool:
        """Engineer features for a specific race"""
        logger.info(f"Engineering features for {race_name}...")
        
        try:
            # Check if cleaned data exists
            race_dir_name = race_name.replace(" ", "_")
            cleaned_file = self.processed_dir / race_dir_name / "laps_cleaned.csv"
            
            if not cleaned_file.exists():
                logger.error(f"❌ Cleaned data not found: {cleaned_file}")
                return False
            
            # Engineer features
            df_features = self.engineer.engineer_features(self.year, race_name)
            
            if df_features is not None and not df_features.empty:
                # Save features
                output_path = self.engineer.save_features(df_features, self.year, race_name)
                logger.info(f"✅ Engineered {df_features.shape[1]} features for {race_name}")
                return True
            else:
                logger.error(f"❌ Feature engineering failed for {race_name}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error engineering features for {race_name}: {e}")
            return False
    
    def prepare_race_model_data(self, race_name: str) -> bool:
        """Prepare model-ready data for a specific race"""
        logger.info(f"Preparing model data for {race_name}...")
        
        try:
            # Check if features exist
            race_dir_name = race_name.replace(" ", "_")
            features_file = self.processed_dir / race_dir_name / "laps_features.csv"
            
            if not features_file.exists():
                logger.error(f"❌ Features file not found: {features_file}")
                return False
            
            # Prepare model data
            result = prepare_model_data(
                year=self.year,
                race_name=race_name,
                target_variable='LapTime',
                test_size=0.2,
                random_state=42
            )
            
            if result:
                logger.info(f"✅ Prepared model data for {race_name}")
                logger.info(f"   - Train samples: {len(result['y_train'])}")
                logger.info(f"   - Test samples: {len(result['y_test'])}")
                logger.info(f"   - Features: {len(result['feature_names'])}")
                return True
            else:
                logger.error(f"❌ Failed to prepare model data for {race_name}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error preparing model data for {race_name}: {e}")
            return False
    
    def process_race_complete_pipeline(self, race_name: str) -> bool:
        """Run complete pipeline for a single race"""
        logger.info(f"🏎️ Starting complete pipeline for {race_name}")
        
        steps = [
            ("Ingest", self.ingest_race_data),
            ("Clean", self.clean_race_data),
            ("Engineer Features", self.engineer_race_features),
            ("Prepare Model Data", self.prepare_race_model_data)
        ]
        
        for step_name, step_func in steps:
            logger.info(f"⚙️ {step_name}: {race_name}")
            if not step_func(race_name):
                logger.error(f"❌ Pipeline failed at {step_name} for {race_name}")
                return False
        
        logger.info(f"✅ Complete pipeline successful for {race_name}")
        return True
    
    def process_multiple_races(self, race_names: List[str]) -> Dict[str, bool]:
        """Process multiple races through the complete pipeline"""
        logger.info(f"🚀 Processing {len(race_names)} races: {', '.join(race_names)}")
        
        results = {}
        successful = 0
        
        for race_name in race_names:
            try:
                success = self.process_race_complete_pipeline(race_name)
                results[race_name] = success
                if success:
                    successful += 1
                    logger.info(f"✅ {race_name}: SUCCESS")
                else:
                    logger.error(f"❌ {race_name}: FAILED")
            except Exception as e:
                logger.error(f"❌ {race_name}: ERROR - {e}")
                results[race_name] = False
        
        logger.info(f"🏁 Pipeline complete: {successful}/{len(race_names)} races successful")
        return results
    
    def generate_summary_report(self, results: Dict[str, bool]) -> str:
        """Generate a summary report of the data ingestion process"""
        successful_races = [race for race, success in results.items() if success]
        failed_races = [race for race, success in results.items() if not success]
        
        report = f"""
# 2025 F1 Data Ingestion Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary
- **Total Races Processed**: {len(results)}
- **Successful**: {len(successful_races)}
- **Failed**: {len(failed_races)}
- **Success Rate**: {len(successful_races)/len(results)*100:.1f}%

## Successful Races
{chr(10).join(f"✅ {race}" for race in successful_races)}

## Failed Races
{chr(10).join(f"❌ {race}" for race in failed_races)}

## Data Locations
- **Raw Data**: `data/raw/{self.year}/`
- **Processed Data**: `data/processed/{self.year}/`
- **Training Data**: `data/processed/{self.year}/[race_name]/train_data.csv`
- **Test Data**: `data/processed/{self.year}/[race_name]/test_data.csv`

## Next Steps
1. Verify data quality in processed directories
2. Run model training on new 2025 datasets
3. Update dashboard to support 2025 data
4. Compare 2025 vs 2024 model performance
"""
        
        # Save report
        report_path = self.logs_dir / f"2025_ingestion_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(report_path, 'w') as f:
            f.write(report)
        
        logger.info(f"📄 Report saved to: {report_path}")
        return report


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Ingest and process 2025 F1 data for ML training")
    parser.add_argument("--year", type=int, default=2025, help="F1 season year")
    parser.add_argument("--races", type=str, help="Comma-separated list of race names")
    parser.add_argument("--all-available", action="store_true", help="Process all available races")
    parser.add_argument("--max-races", type=int, default=10, help="Maximum number of races to process")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = F1DataPipeline2025(year=args.year)
    
    # Determine which races to process
    if args.all_available:
        logger.info("Fetching all available races...")
        available_races = pipeline.get_available_races()
        race_names = [race["race_name"] for race in available_races[:args.max_races]]
    elif args.races:
        race_names = [race.strip() for race in args.races.split(",")]
    else:
        # Default to a few key races
        race_names = [
            "Bahrain Grand Prix",
            "Saudi Arabian Grand Prix", 
            "Australian Grand Prix",
            "Monaco Grand Prix",
            "Austrian Grand Prix"
        ]
    
    logger.info(f"Selected races: {', '.join(race_names)}")
    
    # Process races
    results = pipeline.process_multiple_races(race_names)
    
    # Generate report
    report = pipeline.generate_summary_report(results)
    print("\n" + "="*60)
    print(report)
    print("="*60)
    
    # Return exit code based on success rate
    success_rate = sum(results.values()) / len(results) if results else 0
    if success_rate >= 0.5:  # At least 50% success
        logger.info("🎉 Data ingestion completed successfully!")
        return 0
    else:
        logger.error("❌ Data ingestion had significant failures")
        return 1


if __name__ == "__main__":
    exit(main())
