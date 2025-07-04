"""
F1 API Integration Module

This module provides integration with FastF1 API for F1 data:
- Session data (Practice, Qualifying, Race)
- Lap times and telemetry data
- Driver and constructor information
- Circuit and weather data
- Automatic error handling and caching
"""

import fastf1
import pandas as pd
import numpy as np
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
from pathlib import Path
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enable FastF1 caching
import os
# Get absolute path to cache directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
cache_dir = os.path.join(project_root, 'data', 'fastf1_cache')
os.makedirs(cache_dir, exist_ok=True)
fastf1.Cache.enable_cache(cache_dir)


class F1APIIntegration:
    """
    Main class for F1 API integration using FastF1
    """
    
    def __init__(self, year: int = 2024, cache_dir: str = "data/cache"):
        """
        Initialize F1 API integration
        
        Args:
            year: Championship year to fetch data for
            cache_dir: Directory to store cached data
        """
        self.year = year
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # FastF1 session types
        self.session_types = {
            'FP1': 'Practice 1',
            'FP2': 'Practice 2', 
            'FP3': 'Practice 3',
            'Q': 'Qualifying',
            'S': 'Sprint',
            'SQ': 'Sprint Qualifying',
            'R': 'Race'
        }
        
        # Current year driver and team mappings
        self.driver_mapping = {}
        self.team_mapping = {}
        self.circuit_mapping = {}
        
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path for a given key"""
        return self.cache_dir / f"{cache_key}.json"
    
    def _load_from_cache(self, cache_key: str, max_age_hours: int = 1) -> Optional[Dict]:
        """Load data from cache if it exists and is not expired"""
        cache_path = self._get_cache_path(cache_key)
        
        if cache_path.exists():
            try:
                with open(cache_path, 'r') as f:
                    cached_data = json.load(f)
                
                # Check if cache is still valid
                cache_time = datetime.fromisoformat(cached_data.get('timestamp', '1970-01-01'))
                if datetime.now() - cache_time < timedelta(hours=max_age_hours):
                    logger.info(f"Using cached data for {cache_key}")
                    return cached_data.get('data')
            except Exception as e:
                logger.warning(f"Failed to load cache for {cache_key}: {e}")
        
        return None
    
    def _save_to_cache(self, cache_key: str, data: Any):
        """Save data to cache with timestamp"""
        cache_path = self._get_cache_path(cache_key)
        
        try:
            cache_data = {
                'timestamp': datetime.now().isoformat(),
                'data': data
            }
            with open(cache_path, 'w') as f:
                json.dump(cache_data, f, indent=2, default=str)
            
            logger.info(f"Cached data for {cache_key}")
        except Exception as e:
            logger.warning(f"Failed to cache data for {cache_key}: {e}")
    
    def get_session(self, race_name: str, session_type: str = 'R') -> Optional[Any]:
        """
        Get a FastF1 session for a specific race and session type
        
        Args:
            race_name: Name of the race (e.g., 'Monaco Grand Prix')
            session_type: Type of session ('R', 'Q', 'FP1', 'FP2', 'FP3', 'S', 'SQ')
        """
        cache_key = f"session_{self.year}_{race_name}_{session_type}"
        
        try:
            logger.info(f"Loading session: {self.year} {race_name} {session_type}")
            session = fastf1.get_session(self.year, race_name, session_type)
            session.load()
            return session
            
        except Exception as e:
            logger.error(f"Failed to load session {race_name} {session_type}: {e}")
            return None
    
    def fetch_session_data(self, race_name: str, session_type: str = 'R') -> Optional[Dict]:
        """
        Fetch comprehensive session data using FastF1
        
        Args:
            race_name: Name of the race
            session_type: Type of session
        """
        cache_key = f"session_data_{self.year}_{race_name}_{session_type}"
        
        # Try to load from cache first
        cached_data = self._load_from_cache(cache_key, max_age_hours=24)
        if cached_data:
            return cached_data
        
        session = self.get_session(race_name, session_type)
        if not session:
            return None
        
        try:
            # Get basic session info
            session_data = {
                'year': self.year,
                'race_name': race_name,
                'session_type': session_type,
                'session_name': self.session_types.get(session_type, session_type),
                'circuit_name': session.event.get('EventName', ''),
                'circuit_location': session.event.get('Location', ''),
                'date': str(session.date) if session.date else None,
                'total_laps': len(session.laps) if hasattr(session, 'laps') else 0,
                'drivers': [],
                'laps': [],
                'weather': None
            }
            
            # Get driver information
            if hasattr(session, 'drivers') and session.drivers is not None:
                for driver in session.drivers:
                    driver_info = session.get_driver(driver)
                    if driver_info is not None:
                        session_data['drivers'].append({
                            'driver_number': driver,
                            'driver_name': driver_info.get('FullName', ''),
                            'driver_code': driver_info.get('Abbreviation', ''),
                            'team_name': driver_info.get('TeamName', ''),
                            'team_color': driver_info.get('TeamColor', '')
                        })
            
            # Get lap data
            if hasattr(session, 'laps') and session.laps is not None:
                # Convert laps to serializable format
                laps_df = session.laps
                if not laps_df.empty:
                    # Select key columns and convert to dict
                    lap_columns = ['Driver', 'LapNumber', 'LapTime', 'Sector1Time', 'Sector2Time', 'Sector3Time', 'Compound', 'TyreLife']
                    available_columns = [col for col in lap_columns if col in laps_df.columns]
                    
                    laps_data = laps_df[available_columns].to_dict('records')
                    # Convert timedelta objects to string
                    for lap in laps_data:
                        for key, value in lap.items():
                            if pd.isna(value):
                                lap[key] = None
                            elif hasattr(value, 'total_seconds'):
                                lap[key] = str(value)
                    
                    session_data['laps'] = laps_data[:1000]  # Limit to first 1000 laps for cache
            
            # Get weather data if available
            if hasattr(session, 'weather_data') and session.weather_data is not None:
                weather_df = session.weather_data
                if not weather_df.empty:
                    # Get average weather conditions
                    weather_cols = ['AirTemp', 'TrackTemp', 'Humidity', 'Pressure', 'WindSpeed', 'WindDirection']
                    available_weather = [col for col in weather_cols if col in weather_df.columns]
                    
                    if available_weather:
                        weather_means = weather_df[available_weather].mean()
                        session_data['weather'] = weather_means.to_dict()
            
            # Cache the data
            self._save_to_cache(cache_key, session_data)
            
            logger.info(f"Successfully fetched session data for {race_name} {session_type}")
            return session_data
            
        except Exception as e:
            logger.error(f"Failed to process session data: {e}")
            return None
    
    def fetch_race_schedule(self) -> Optional[List[Dict]]:
        """
        Fetch race schedule using FastF1
        """
        cache_key = f"schedule_{self.year}"
        
        # Try to load from cache first
        cached_data = self._load_from_cache(cache_key, max_age_hours=24)
        if cached_data:
            return cached_data
        
        try:
            schedule = fastf1.get_event_schedule(self.year)
            races = []
            
            for idx, event in schedule.iterrows():
                race_data = {
                    'round': int(event.get('RoundNumber', idx + 1)),
                    'race_name': str(event.get('EventName', '')),
                    'circuit_name': str(event.get('Location', '')),
                    'country': str(event.get('Country', '')),
                    'date': str(event.get('EventDate', '')),
                    'format': str(event.get('EventFormat', '')),
                    'session1_date': str(event.get('Session1Date', '')),
                    'session2_date': str(event.get('Session2Date', '')),
                    'session3_date': str(event.get('Session3Date', '')),
                    'session4_date': str(event.get('Session4Date', '')),
                    'session5_date': str(event.get('Session5Date', ''))
                }
                races.append(race_data)
            
            # Cache the data
            self._save_to_cache(cache_key, races)
            
            logger.info(f"Successfully fetched {len(races)} races for {self.year}")
            return races
            
        except Exception as e:
            logger.error(f"Failed to fetch race schedule using FastF1: {e}")
            return None
    
    def fetch_lap_data(self, race_name: str, session_type: str = 'R') -> Optional[pd.DataFrame]:
        """
        Fetch lap data for a specific race session using FastF1
        
        Args:
            race_name: Name of the race
            session_type: Type of session ('R', 'Q', 'FP1', 'FP2', 'FP3', 'S', 'SQ')
        """
        session = self.get_session(race_name, session_type)
        if not session:
            return None
        
        try:
            return session.laps
        except Exception as e:
            logger.error(f"Failed to fetch lap data: {e}")
            return None
    
    def fetch_telemetry_data(self, race_name: str, session_type: str = 'R', driver: str = None) -> Optional[pd.DataFrame]:
        """
        Fetch telemetry data for a specific driver and session
        
        Args:
            race_name: Name of the race
            session_type: Type of session
            driver: Driver identifier (abbreviation, number, or name)
        """
        session = self.get_session(race_name, session_type)
        if not session:
            return None
        
        try:
            if driver:
                # Get specific driver's fastest lap telemetry
                driver_laps = session.laps.pick_driver(driver)
                if not driver_laps.empty:
                    fastest_lap = driver_laps.pick_fastest()
                    return fastest_lap.get_telemetry()
            else:
                # Get all telemetry data
                return session.laps.get_telemetry()
                
        except Exception as e:
            logger.error(f"Failed to fetch telemetry data: {e}")
            return None
    
    def get_driver_info(self, race_name: str = None, session_type: str = 'R') -> Optional[Dict]:
        """
        Fetch driver information using FastF1
        
        Args:
            race_name: Optional race name to get driver info from a specific session
            session_type: Type of session
        """
        cache_key = f"drivers_{self.year}_{race_name or 'all'}_{session_type}"
        
        # Try to load from cache first
        cached_data = self._load_from_cache(cache_key, max_age_hours=24)
        if cached_data:
            return cached_data
        
        try:
            drivers = {}
            
            if race_name:
                # Get driver info from specific session
                session = self.get_session(race_name, session_type)
                if session and hasattr(session, 'drivers'):
                    for driver_num in session.drivers:
                        driver_info = session.get_driver(driver_num)
                        if driver_info is not None:
                            drivers[driver_num] = {
                                'driver_number': driver_num,
                                'driver_name': driver_info.get('FullName', ''),
                                'driver_code': driver_info.get('Abbreviation', ''),
                                'team_name': driver_info.get('TeamName', ''),
                                'team_color': driver_info.get('TeamColor', ''),
                                'country_code': driver_info.get('CountryCode', ''),
                                'first_name': driver_info.get('FirstName', ''),
                                'last_name': driver_info.get('LastName', '')
                            }
            else:
                # Get all drivers from the season schedule
                schedule = self.fetch_race_schedule()
                if schedule:
                    # Get drivers from the first available race
                    for race in schedule:
                        session = self.get_session(race['race_name'], 'R')
                        if session and hasattr(session, 'drivers'):
                            for driver_num in session.drivers:
                                driver_info = session.get_driver(driver_num)
                                if driver_info is not None:
                                    drivers[driver_num] = {
                                        'driver_number': driver_num,
                                        'driver_name': driver_info.get('FullName', ''),
                                        'driver_code': driver_info.get('Abbreviation', ''),
                                        'team_name': driver_info.get('TeamName', ''),
                                        'team_color': driver_info.get('TeamColor', ''),
                                        'country_code': driver_info.get('CountryCode', ''),
                                        'first_name': driver_info.get('FirstName', ''),
                                        'last_name': driver_info.get('LastName', '')
                                    }
                            break  # Only need one race to get all drivers
            
            # Cache the data
            self._save_to_cache(cache_key, drivers)
            
            logger.info(f"Successfully fetched info for {len(drivers)} drivers")
            return drivers
            
        except Exception as e:
            logger.error(f"Failed to fetch driver info: {e}")
            return None
    
    def fetch_completed_races_data(self, races: List[Dict]) -> Optional[Dict]:
        """
        Fetch race data for completed races to provide to season analysis
        
        Args:
            races: List of race dictionaries from schedule
        """
        races_data = {}
        
        for race in races:
            race_name = race['race_name']
            
            # Try to get race session data
            try:
                laps_df = self.fetch_lap_data(race_name, 'R')
                if laps_df is not None and not laps_df.empty:
                    # Convert to format expected by season_analysis
                    race_data = []
                    
                    for _, lap in laps_df.iterrows():
                        # Map driver number to sequential ID
                        driver_num = lap.get('DriverNumber', 0)
                        driver_id = driver_num % 20  # Simple mapping for demo
                        
                        # Convert lap time to seconds if it's a timedelta
                        lap_time_seconds = 0
                        if pd.notna(lap.get('LapTime')):
                            lap_time = lap['LapTime']
                            if hasattr(lap_time, 'total_seconds'):
                                lap_time_seconds = lap_time.total_seconds()
                            else:
                                lap_time_seconds = 80.0  # fallback
                        else:
                            lap_time_seconds = 80.0  # fallback for missing data
                        
                        race_data.append({
                            'Driver': driver_id,
                            'Team': driver_id // 2,  # Two drivers per team
                            'LapNumber': int(lap.get('LapNumber', 1)),
                            'LapTime': lap_time_seconds,
                            'Position': int(lap.get('Position', 20)) if pd.notna(lap.get('Position')) else 20,
                            'TyreLife': int(lap.get('TyreLife', 10)) if pd.notna(lap.get('TyreLife')) else 10
                        })
                    
                    if race_data:
                        races_data[race_name] = pd.DataFrame(race_data)
                        logger.info(f"Successfully fetched race data for {race_name}")
                    
            except Exception as e:
                logger.warning(f"Failed to fetch race data for {race_name}: {str(e)}")
                continue
        
        return races_data if races_data else None
    
    def create_fallback_data(self) -> Dict:
        """
        Create fallback data when APIs are unavailable
        """
        logger.info("Creating fallback data for API failure")
        
        # Create realistic 2024 season data
        fallback_drivers = [
            {'position': 1, 'points': 575, 'wins': 19, 'driver_name': 'Max Verstappen', 'team_name': 'Red Bull Racing Honda RBPT'},
            {'position': 2, 'points': 285, 'wins': 2, 'driver_name': 'Lando Norris', 'team_name': 'McLaren Mercedes'},
            {'position': 3, 'points': 291, 'wins': 0, 'driver_name': 'Charles Leclerc', 'team_name': 'Ferrari'},
            {'position': 4, 'points': 238, 'wins': 1, 'driver_name': 'Oscar Piastri', 'team_name': 'McLaren Mercedes'},
            {'position': 5, 'points': 190, 'wins': 0, 'driver_name': 'Carlos Sainz', 'team_name': 'Ferrari'},
            {'position': 6, 'points': 177, 'wins': 0, 'driver_name': 'George Russell', 'team_name': 'Mercedes'},
            {'position': 7, 'points': 164, 'wins': 2, 'driver_name': 'Lewis Hamilton', 'team_name': 'Mercedes'},
            {'position': 8, 'points': 152, 'wins': 0, 'driver_name': 'Sergio Perez', 'team_name': 'Red Bull Racing Honda RBPT'},
            {'position': 9, 'points': 86, 'wins': 0, 'driver_name': 'Fernando Alonso', 'team_name': 'Aston Martin Aramco Mercedes'},
            {'position': 10, 'points': 49, 'wins': 0, 'driver_name': 'Nico Hulkenberg', 'team_name': 'Haas Ferrari'}
        ]
        
        fallback_constructors = [
            {'position': 1, 'points': 727, 'wins': 19, 'team_name': 'Red Bull Racing Honda RBPT'},
            {'position': 2, 'points': 523, 'wins': 3, 'team_name': 'McLaren Mercedes'},
            {'position': 3, 'points': 529, 'wins': 0, 'team_name': 'Ferrari'},
            {'position': 4, 'points': 341, 'wins': 2, 'team_name': 'Mercedes'},
            {'position': 5, 'points': 94, 'wins': 0, 'team_name': 'Aston Martin Aramco Mercedes'}
        ]
        
        return {
            'drivers': fallback_drivers,
            'constructors': fallback_constructors,
            'season': self.year,
            'round': 22,
            'is_fallback': True
        }
    
    def get_comprehensive_season_data(self) -> Dict:
        """
        Fetch comprehensive season data using FastF1
        """
        logger.info(f"Fetching comprehensive F1 data for {self.year}")
        
        season_data = {
            'year': self.year,
            'last_updated': datetime.now().isoformat(),
            'data_sources': [],
            'schedule': None,
            'drivers': None,
            'races_data': {},
            'is_fallback': False
        }
        
        try:
            # Fetch race schedule
            schedule = self.fetch_race_schedule()
            if schedule:
                season_data['schedule'] = schedule
                season_data['data_sources'].append('fastf1_schedule')
                logger.info("✅ Successfully fetched race schedule")
            else:
                logger.warning("❌ Failed to fetch race schedule")
            
            # Fetch driver info
            drivers = self.get_driver_info()
            if drivers:
                season_data['drivers'] = drivers
                season_data['data_sources'].append('fastf1_drivers')
                logger.info("✅ Successfully fetched driver info")
            else:
                logger.warning("❌ Failed to fetch driver info")
            
            # Fetch race data for completed races
            if schedule:
                races_data = self.fetch_completed_races_data(schedule[:5])  # First 5 races for testing
                if races_data:
                    season_data['races_data'] = races_data
                    season_data['data_sources'].append('fastf1_races')
                    logger.info("✅ Successfully fetched race data")
            
            # If no data was fetched successfully, use fallback
            if not season_data['data_sources']:
                logger.warning("No real data available, using fallback data")
                fallback_data = self.create_fallback_data()
                season_data['standings'] = fallback_data
                season_data['is_fallback'] = True
                season_data['data_sources'].append('fallback')
            
            logger.info(f"Data collection complete. Sources used: {', '.join(season_data['data_sources'])}")
            return season_data
            
        except Exception as e:
            logger.error(f"Failed to fetch comprehensive season data: {e}")
            # Return fallback data on any error
            fallback_data = self.create_fallback_data()
            season_data['standings'] = fallback_data
            season_data['is_fallback'] = True
            season_data['data_sources'] = ['fallback']
            return season_data
    
    def save_api_data_to_pipeline_format(self, season_data: Dict, output_dir: str = "data/api_cache"):
        """
        Save API data in a format compatible with the existing pipeline
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save raw API data
        api_data_path = output_path / f"f1_api_data_{self.year}.json"
        with open(api_data_path, 'w') as f:
            json.dump(season_data, f, indent=2, default=str)
        
        logger.info(f"API data saved to: {api_data_path}")
        
        # Convert to pipeline format if we have standings data
        if season_data.get('standings'):
            self._convert_to_pipeline_format(season_data, output_path)
    
    def _convert_to_pipeline_format(self, season_data: Dict, output_path: Path):
        """
        Convert API data to format compatible with existing season_analysis.py
        """
        try:
            standings = season_data['standings']
            
            # Create driver mapping
            driver_mapping = {}
            team_mapping = {}
            
            for i, driver in enumerate(standings['drivers']):
                driver_mapping[i] = driver['driver_name'].split()[-1]  # Use last name
                # Map every two drivers to a team
                team_id = i // 2
                if team_id not in team_mapping:
                    team_mapping[team_id] = driver['team_name']
            
            # Create pipeline-compatible data structure
            pipeline_data = {
                'year': self.year,
                'driver_mapping': driver_mapping,
                'team_mapping': team_mapping,
                'standings': {
                    'drivers': standings['drivers'],
                    'constructors': standings.get('constructors', [])
                },
                'metadata': {
                    'last_updated': season_data['last_updated'],
                    'data_sources': season_data['data_sources'],
                    'is_fallback': season_data.get('is_fallback', False)
                }
            }
            
            # Save pipeline-compatible format
            pipeline_path = output_path / f"pipeline_data_{self.year}.json"
            with open(pipeline_path, 'w') as f:
                json.dump(pipeline_data, f, indent=2)
            
            logger.info(f"Pipeline-compatible data saved to: {pipeline_path}")
            
        except Exception as e:
            logger.error(f"Failed to convert to pipeline format: {e}")


def main():
    """
    Main function to demonstrate API integration
    """
    print("🏎️ F1 API Integration Starting...")
    print("=" * 50)
    
    # Initialize API integration for current year (2024, as 2025 season hasn't started)
    api = F1APIIntegration(year=2024)
    
    try:
        # Fetch comprehensive season data
        season_data = api.get_comprehensive_season_data()
        
        print(f"\n📊 Data Sources Used: {', '.join(season_data['data_sources'])}")
        print(f"🔄 Last Updated: {season_data['last_updated']}")
        print(f"⚠️ Using Fallback Data: {season_data.get('is_fallback', False)}")
        
        # Display current standings if available
        if season_data.get('standings') and season_data['standings'].get('drivers'):
            print("\n🏁 Current Driver Standings (Top 10):")
            for i, driver in enumerate(season_data['standings']['drivers'][:10]):
                print(f"  {driver['position']}. {driver['driver_name']} - {driver['points']} points ({driver['wins']} wins)")
        
        # Display constructor standings if available
        if season_data.get('standings') and season_data['standings'].get('constructors'):
            print("\n🏭 Constructor Standings (Top 5):")
            for constructor in season_data['standings']['constructors'][:5]:
                print(f"  {constructor['position']}. {constructor['team_name']} - {constructor['points']} points")
        
        # Save data for use by other pipeline components
        api.save_api_data_to_pipeline_format(season_data)
        
        print("\n✅ API Integration Complete!")
        print("📁 Data saved to data/api_cache/ directory")
        print("🔗 Ready for integration with season_analysis.py")
        
    except Exception as e:
        print(f"❌ API integration failed: {e}")
        logger.exception("Full error details:")


if __name__ == "__main__":
    main()
