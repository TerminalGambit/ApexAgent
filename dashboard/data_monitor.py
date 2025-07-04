"""
Data Monitoring Utility for F1 ML Dashboard

Scans and analyzes all available data across seasons (2024, 2025)
and provides comprehensive data quality metrics and statistics.
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import glob


class F1DataMonitor:
    """Monitor and analyze all F1 data across seasons"""
    
    def __init__(self, base_dir="../data"):
        self.base_dir = Path(base_dir)
        self.seasons = [2024, 2025]
        self.data_summary = {}
        
    def scan_all_data(self):
        """Scan all available data across seasons"""
        print("🔍 Scanning F1 data across all seasons...")
        
        summary = {
            'scan_timestamp': datetime.now().isoformat(),
            'seasons': {},
            'total_stats': {
                'total_races': 0,
                'total_laps': 0,
                'total_drivers': 0,
                'total_features': 0,
                'data_size_mb': 0
            }
        }
        
        for year in self.seasons:
            summary['seasons'][year] = self.scan_season_data(year)
            
        # Calculate total stats
        for year_data in summary['seasons'].values():
            if year_data:
                summary['total_stats']['total_races'] += year_data.get('total_races', 0)
                summary['total_stats']['total_laps'] += year_data.get('total_laps', 0)
                summary['total_stats']['data_size_mb'] += year_data.get('data_size_mb', 0)
        
        self.data_summary = summary
        return summary
    
    def scan_season_data(self, year):
        """Scan data for a specific season"""
        print(f"  📅 Scanning {year} season data...")
        
        season_data = {
            'year': year,
            'races': {},
            'total_races': 0,
            'total_laps': 0,
            'data_size_mb': 0,
            'data_quality': {
                'raw_data_available': 0,
                'cleaned_data_available': 0,
                'features_available': 0,
                'model_ready_available': 0
            }
        }
        
        # Check different data directories
        directories_to_check = [
            self.base_dir / "raw" / str(year),
            self.base_dir / "processed" / str(year),
            self.base_dir / "analysis" / str(year),
            self.base_dir / "predictions" / str(year)
        ]
        
        # Scan processed data (most complete)
        processed_dir = self.base_dir / "processed" / str(year)
        if processed_dir.exists():
            for race_dir in processed_dir.iterdir():
                if race_dir.is_dir():
                    race_name = race_dir.name.replace("_", " ")
                    race_data = self.analyze_race_data(race_dir, year, race_name)
                    if race_data:
                        season_data['races'][race_name] = race_data
                        season_data['total_races'] += 1
                        season_data['total_laps'] += race_data.get('total_laps', 0)
        
        # Scan raw data for additional races
        raw_dir = self.base_dir / "raw" / str(year)
        if raw_dir.exists():
            for race_dir in raw_dir.iterdir():
                if race_dir.is_dir():
                    race_name = race_dir.name.replace("_", " ")
                    if race_name not in season_data['races']:
                        race_data = self.analyze_raw_race_data(race_dir, year, race_name)
                        if race_data:
                            season_data['races'][race_name] = race_data
                            season_data['total_races'] += 1
        
        # Calculate data quality metrics
        for race_data in season_data['races'].values():
            if race_data.get('files', {}).get('raw_data'):
                season_data['data_quality']['raw_data_available'] += 1
            if race_data.get('files', {}).get('cleaned_data'):
                season_data['data_quality']['cleaned_data_available'] += 1
            if race_data.get('files', {}).get('features_data'):
                season_data['data_quality']['features_available'] += 1
            if race_data.get('files', {}).get('model_ready'):
                season_data['data_quality']['model_ready_available'] += 1
        
        # Calculate total data size
        season_data['data_size_mb'] = self.calculate_directory_size(
            self.base_dir / "processed" / str(year)
        ) + self.calculate_directory_size(
            self.base_dir / "raw" / str(year)
        )
        
        return season_data
    
    def analyze_race_data(self, race_dir, year, race_name):
        """Analyze processed race data directory"""
        race_data = {
            'race_name': race_name,
            'year': year,
            'total_laps': 0,
            'total_features': 0,
            'train_samples': 0,
            'test_samples': 0,
            'files': {
                'raw_data': False,
                'cleaned_data': False,
                'features_data': False,
                'model_ready': False
            },
            'file_sizes': {},
            'data_quality_score': 0,
            'last_modified': None
        }
        
        try:
            # Check for different file types
            files_to_check = {
                'laps_cleaned.csv': 'cleaned_data',
                'laps_features.csv': 'features_data',
                'train_data.csv': 'model_ready',
                'test_data.csv': 'model_ready',
                'feature_names.csv': 'model_ready'
            }
            
            for filename, data_type in files_to_check.items():
                file_path = race_dir / filename
                if file_path.exists():
                    race_data['files'][data_type] = True
                    race_data['file_sizes'][filename] = file_path.stat().st_size / (1024 * 1024)  # MB
                    
                    # Update last modified time
                    modified_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if not race_data['last_modified'] or modified_time > datetime.fromisoformat(race_data['last_modified'] if race_data['last_modified'] else '1970-01-01T00:00:00'):
                        race_data['last_modified'] = modified_time.isoformat()
            
            # Analyze features data if available
            features_file = race_dir / "laps_features.csv"
            if features_file.exists():
                try:
                    df = pd.read_csv(features_file, nrows=5)  # Sample for speed
                    race_data['total_features'] = len(df.columns)
                    
                    # Get total row count more efficiently
                    with open(features_file, 'r') as f:
                        race_data['total_laps'] = sum(1 for line in f) - 1  # Subtract header
                except Exception as e:
                    print(f"    ⚠️ Error reading {features_file}: {e}")
            
            # Analyze train/test data if available
            train_file = race_dir / "train_data.csv"
            test_file = race_dir / "test_data.csv"
            
            if train_file.exists():
                try:
                    with open(train_file, 'r') as f:
                        race_data['train_samples'] = sum(1 for line in f) - 1
                except Exception:
                    pass
            
            if test_file.exists():
                try:
                    with open(test_file, 'r') as f:
                        race_data['test_samples'] = sum(1 for line in f) - 1
                except Exception:
                    pass
            
            # Calculate data quality score (0-100)
            quality_factors = [
                race_data['files']['cleaned_data'],
                race_data['files']['features_data'],
                race_data['files']['model_ready'],
                race_data['total_laps'] > 0,
                race_data['total_features'] > 10
            ]
            race_data['data_quality_score'] = sum(quality_factors) / len(quality_factors) * 100
            
        except Exception as e:
            print(f"    ❌ Error analyzing {race_name}: {e}")
            return None
        
        return race_data
    
    def analyze_raw_race_data(self, race_dir, year, race_name):
        """Analyze raw race data directory"""
        race_data = {
            'race_name': race_name,
            'year': year,
            'total_laps': 0,
            'total_features': 0,
            'train_samples': 0,
            'test_samples': 0,
            'files': {
                'raw_data': False,
                'cleaned_data': False,
                'features_data': False,
                'model_ready': False
            },
            'file_sizes': {},
            'data_quality_score': 0,
            'last_modified': None,
            'status': 'Raw data only'
        }
        
        try:
            # Check for raw lap data
            laps_file = race_dir / "laps.csv"
            if laps_file.exists():
                race_data['files']['raw_data'] = True
                race_data['file_sizes']['laps.csv'] = laps_file.stat().st_size / (1024 * 1024)
                
                # Quick lap count
                try:
                    with open(laps_file, 'r') as f:
                        race_data['total_laps'] = sum(1 for line in f) - 1
                except Exception:
                    pass
                
                modified_time = datetime.fromtimestamp(laps_file.stat().st_mtime)
                race_data['last_modified'] = modified_time.isoformat()
            
            # Check for session metadata
            metadata_file = race_dir / "session_metadata.json"
            if metadata_file.exists():
                race_data['file_sizes']['session_metadata.json'] = metadata_file.stat().st_size / (1024 * 1024)
            
            # Raw data quality score
            race_data['data_quality_score'] = 25 if race_data['files']['raw_data'] else 0
            
        except Exception as e:
            print(f"    ❌ Error analyzing raw {race_name}: {e}")
            return None
        
        return race_data
    
    def calculate_directory_size(self, directory):
        """Calculate total size of directory in MB"""
        if not directory.exists():
            return 0
        
        total_size = 0
        try:
            for file_path in directory.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
        except Exception:
            pass
        
        return total_size / (1024 * 1024)  # Convert to MB
    
    def get_data_pipeline_status(self):
        """Get status of data pipeline stages"""
        if not self.data_summary:
            self.scan_all_data()
        
        pipeline_status = {}
        
        for year, year_data in self.data_summary['seasons'].items():
            if not year_data:
                continue
                
            status = {
                'total_races': year_data['total_races'],
                'pipeline_stages': {
                    'raw_ingestion': 0,
                    'data_cleaning': 0,
                    'feature_engineering': 0,
                    'model_preparation': 0
                },
                'completion_percentage': 0
            }
            
            total_races = year_data['total_races']
            if total_races > 0:
                quality = year_data['data_quality']
                status['pipeline_stages']['raw_ingestion'] = quality['raw_data_available'] / total_races * 100
                status['pipeline_stages']['data_cleaning'] = quality['cleaned_data_available'] / total_races * 100
                status['pipeline_stages']['feature_engineering'] = quality['features_available'] / total_races * 100
                status['pipeline_stages']['model_preparation'] = quality['model_ready_available'] / total_races * 100
                
                # Overall completion is the average of all stages
                status['completion_percentage'] = sum(status['pipeline_stages'].values()) / 4
            
            pipeline_status[year] = status
        
        return pipeline_status
    
    def get_feature_analysis(self):
        """Analyze features across all races"""
        feature_analysis = {
            'total_unique_features': set(),
            'feature_consistency': {},
            'average_features_per_race': 0,
            'races_analyzed': 0
        }
        
        for year_data in self.data_summary['seasons'].values():
            if not year_data:
                continue
                
            for race_data in year_data['races'].values():
                if race_data.get('total_features', 0) > 0:
                    feature_analysis['races_analyzed'] += 1
                    
                    # Try to get actual feature names
                    race_dir = self.base_dir / "processed" / str(race_data['year']) / race_data['race_name'].replace(" ", "_")
                    feature_names_file = race_dir / "feature_names.csv"
                    
                    if feature_names_file.exists():
                        try:
                            features_df = pd.read_csv(feature_names_file)
                            features = set(features_df['feature'].tolist())
                            feature_analysis['total_unique_features'].update(features)
                            
                            # Track feature consistency
                            for feature in features:
                                if feature not in feature_analysis['feature_consistency']:
                                    feature_analysis['feature_consistency'][feature] = 0
                                feature_analysis['feature_consistency'][feature] += 1
                                
                        except Exception:
                            pass
        
        # Calculate averages
        if feature_analysis['races_analyzed'] > 0:
            total_features = sum(race_data.get('total_features', 0) 
                               for year_data in self.data_summary['seasons'].values() 
                               if year_data
                               for race_data in year_data['races'].values())
            feature_analysis['average_features_per_race'] = total_features / feature_analysis['races_analyzed']
        
        # Convert set to list for JSON serialization
        feature_analysis['total_unique_features'] = list(feature_analysis['total_unique_features'])
        
        return feature_analysis
    
    def save_monitoring_report(self, output_file="data_monitoring_report.json"):
        """Save comprehensive monitoring report"""
        if not self.data_summary:
            self.scan_all_data()
        
        report = {
            'data_summary': self.data_summary,
            'pipeline_status': self.get_data_pipeline_status(),
            'feature_analysis': self.get_feature_analysis(),
            'report_metadata': {
                'generated_at': datetime.now().isoformat(),
                'total_data_size_mb': self.data_summary['total_stats']['data_size_mb'],
                'monitoring_version': '1.0'
            }
        }
        
        output_path = Path(output_file)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"📄 Monitoring report saved to: {output_path}")
        return report


def main():
    """Main function for standalone execution"""
    print("🏎️ F1 Data Monitor - Scanning all available data...")
    
    monitor = F1DataMonitor()
    report = monitor.save_monitoring_report("../dashboard/data_monitoring_report.json")
    
    print("\n📊 Data Summary:")
    print(f"  Total Races: {report['data_summary']['total_stats']['total_races']}")
    print(f"  Total Laps: {report['data_summary']['total_stats']['total_laps']:,}")
    print(f"  Data Size: {report['data_summary']['total_stats']['data_size_mb']:.1f} MB")
    
    print("\n🔧 Pipeline Status:")
    for year, status in report['pipeline_status'].items():
        print(f"  {year}: {status['completion_percentage']:.1f}% complete ({status['total_races']} races)")
    
    print("\n✅ Monitoring complete!")


if __name__ == "__main__":
    main()
