import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import json


class SeasonStandingsAnalyzer:
    """
    Analyze current season standings and driver performance trends
    """
    
    def __init__(self, year=2025):
        self.year = year
        self.data_dir = f"data/processed/{year}/"
        self.output_dir = f"data/analysis/{year}/"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 2025 Driver mappings with championship-leading Piastri
        self.driver_mapping = {
            0: "Piastri", 1: "Norris", 2: "Verstappen", 3: "Perez", 
            4: "Leclerc", 5: "Hamilton", 6: "Russell", 7: "Sainz",
            8: "Alonso", 9: "Stroll", 10: "Ocon", 11: "Gasly",
            12: "Albon", 13: "Colapinto", 14: "Tsunoda", 15: "Lawson",
            16: "Hulkenberg", 17: "Magnussen", 18: "Bottas", 19: "Zhou"
        }
        
        self.team_mapping = {
            0: "McLaren", 1: "Red Bull", 2: "Ferrari", 3: "Mercedes",
            4: "Aston Martin", 5: "Alpine", 6: "Williams", 7: "RB",
            8: "Haas", 9: "Sauber"
        }
        
    def load_race_data(self):
        """Load available race data from the season"""
        print("Loading season race data...")
        
        # For now, we'll use Monaco data and simulate additional races
        try:
            monaco_data = pd.read_csv(f"{self.data_dir}Monaco Grand Prix/laps_features_clean.csv")
            
            # Create simulated race results based on Monaco performance
            self.races_data = {
                'Monaco Grand Prix': monaco_data
            }
            
            # Simulate additional races for demonstration
            self._simulate_season_races(monaco_data)
            
            print(f"Loaded {len(self.races_data)} races for analysis")
            
        except Exception as e:
            print(f"Error loading race data: {e}")
            self._create_dummy_season_data()
    
    def _simulate_season_races(self, base_data):
        """Simulate additional races based on Monaco data"""
        race_calendar = [
            'Bahrain Grand Prix', 'Saudi Arabian Grand Prix', 'Australian Grand Prix',
            'Japanese Grand Prix', 'Chinese Grand Prix', 'Miami Grand Prix',
            'Emilia Romagna Grand Prix', 'Monaco Grand Prix', 'Canadian Grand Prix',
            'Spanish Grand Prix', 'Austrian Grand Prix', 'British Grand Prix'
        ]
        
        for race in race_calendar:
            if race not in self.races_data:
                # Create variation of Monaco data with some randomness
                simulated_data = base_data.copy()
                
                # Add track-specific variations
                track_factor = np.random.normal(1.0, 0.05, len(simulated_data))
                simulated_data['LapTime'] = simulated_data['LapTime'] * track_factor
                
                # Vary positions slightly
                position_noise = np.random.normal(0, 1, len(simulated_data))
                simulated_data['Position'] = np.clip(
                    simulated_data['Position'] + position_noise, 1, 20
                ).round().astype(int)
                
                self.races_data[race] = simulated_data
    
    def _create_dummy_season_data(self):
        """Create realistic 2025 season data with Piastri leading"""
        print("Creating realistic 2025 season data...")
        
        # 2025 race calendar up to Silverstone
        races = [
            'Bahrain GP', 'Saudi Arabia GP', 'Australia GP', 'Japan GP', 
            'China GP', 'Miami GP', 'Imola GP', 'Monaco GP', 'Canada GP', 
            'Spain GP', 'Austria GP'
        ]
        
        # Realistic 2025 performance hierarchy with Piastri leading
        driver_performance = {
            0: 0.95,   # Piastri (McLaren) - Championship leader
            1: 0.97,   # Norris (McLaren) - Strong teammate
            2: 1.02,   # Verstappen (Red Bull) - Slightly off pace this year
            3: 1.05,   # Perez (Red Bull)
            4: 0.98,   # Leclerc (Ferrari) - Competitive
            5: 1.01,   # Hamilton (Mercedes) - Experience counts
            6: 1.03,   # Russell (Mercedes)
            7: 1.04,   # Sainz (Williams) - New team
            8: 1.06,   # Alonso (Aston Martin)
            9: 1.08,   # Stroll (Aston Martin)
            10: 1.07,  # Ocon (Alpine)
            11: 1.09,  # Gasly (Alpine)
            12: 1.10,  # Albon (Williams)
            13: 1.15,  # Colapinto (Williams) - Rookie
            14: 1.12,  # Tsunoda (RB)
            15: 1.14,  # Lawson (RB) - Rookie
            16: 1.13,  # Hulkenberg (Haas)
            17: 1.16,  # Magnussen (Haas)
            18: 1.17,  # Bottas (Sauber)
            19: 1.18   # Zhou (Sauber)
        }
        
        self.races_data = {}
        
        for race_idx, race in enumerate(races):
            n_laps = np.random.randint(55, 75)
            data = []
            
            # Start with realistic grid positions based on performance
            grid_positions = list(range(20))
            # Add some qualifying variation
            np.random.shuffle(grid_positions)
            
            for driver in range(20):
                start_pos = grid_positions.index(driver) + 1
                current_pos = start_pos
                
                for lap in range(1, n_laps + 1):
                    # Base lap time with driver performance factor
                    base_time = 78.0  # Base lap time in seconds
                    driver_factor = driver_performance[driver]
                    
                    lap_time = base_time * driver_factor + np.random.normal(0, 0.5)
                    
                    # Simulate position changes during race
                    if lap > 10:  # After first 10 laps, positions settle
                        # Better drivers gradually move up
                        if driver_factor < 1.0 and current_pos > driver:
                            if np.random.random() < 0.1:  # 10% chance per lap
                                current_pos = max(1, current_pos - 1)
                        elif driver_factor > 1.0 and current_pos < driver + 5:
                            if np.random.random() < 0.05:  # 5% chance per lap
                                current_pos = min(20, current_pos + 1)
                    
                    data.append({
                        'Driver': driver,
                        'Team': driver // 2,
                        'LapNumber': lap,
                        'LapTime': lap_time,
                        'Position': current_pos,
                        'TyreLife': min(50, lap + np.random.randint(-3, 3))
                    })
            
            self.races_data[race] = pd.DataFrame(data)
    
    def calculate_driver_standings(self):
        """Calculate current championship standings"""
        print("Calculating driver championship standings...")
        
        # Points system (1st=25, 2nd=18, 3rd=15, 4th=12, 5th=10, 6th=8, 7th=6, 8th=4, 9th=2, 10th=1)
        points_system = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
        
        standings = {}
        
        for race_name, race_data in self.races_data.items():
            # Get final positions for each driver
            final_positions = race_data.groupby('Driver')['Position'].last()
            
            for driver, position in final_positions.items():
                driver_name = self.driver_mapping.get(driver, f"Driver_{driver}")
                
                if driver_name not in standings:
                    standings[driver_name] = {
                        'points': 0,
                        'races': 0,
                        'wins': 0,
                        'podiums': 0,
                        'points_per_race': 0,
                        'avg_position': 0,
                        'positions': []
                    }
                
                # Award points
                points = points_system.get(int(position), 0)
                standings[driver_name]['points'] += points
                standings[driver_name]['races'] += 1
                standings[driver_name]['positions'].append(position)
                
                # Track wins and podiums
                if position == 1:
                    standings[driver_name]['wins'] += 1
                if position <= 3:
                    standings[driver_name]['podiums'] += 1
        
        # Calculate averages
        for driver in standings:
            races = standings[driver]['races']
            if races > 0:
                standings[driver]['points_per_race'] = standings[driver]['points'] / races
                standings[driver]['avg_position'] = np.mean(standings[driver]['positions'])
        
        # Convert to DataFrame and sort by points
        standings_df = pd.DataFrame.from_dict(standings, orient='index')
        standings_df = standings_df.sort_values('points', ascending=False).reset_index()
        standings_df.rename(columns={'index': 'Driver'}, inplace=True)
        standings_df['Championship_Position'] = range(1, len(standings_df) + 1)
        
        self.driver_standings = standings_df
        
        print(f"Championship standings calculated for {len(standings_df)} drivers")
        return standings_df
    
    def calculate_team_standings(self):
        """Calculate constructor championship standings"""
        print("Calculating constructor championship standings...")
        
        team_standings = {}
        
        # Aggregate driver points by team
        for _, driver_row in self.driver_standings.iterrows():
            driver_name = driver_row['Driver']
            points = driver_row['points']
            
            # Find team for this driver (simplified mapping)
            team_id = None
            for driver_id, name in self.driver_mapping.items():
                if name == driver_name:
                    team_id = driver_id // 2
                    break
            
            if team_id is not None:
                team_name = self.team_mapping.get(team_id, f"Team_{team_id}")
                
                if team_name not in team_standings:
                    team_standings[team_name] = {
                        'points': 0,
                        'drivers': [],
                        'wins': 0,
                        'podiums': 0
                    }
                
                team_standings[team_name]['points'] += points
                team_standings[team_name]['drivers'].append(driver_name)
                team_standings[team_name]['wins'] += driver_row['wins']
                team_standings[team_name]['podiums'] += driver_row['podiums']
        
        # Convert to DataFrame
        teams_df = pd.DataFrame.from_dict(team_standings, orient='index')
        teams_df = teams_df.sort_values('points', ascending=False).reset_index()
        teams_df.rename(columns={'index': 'Team'}, inplace=True)
        teams_df['Championship_Position'] = range(1, len(teams_df) + 1)
        
        self.team_standings = teams_df
        
        print(f"Constructor standings calculated for {len(teams_df)} teams")
        return teams_df
    
    def analyze_driver_performance_trends(self):
        """Analyze driver performance trends throughout the season"""
        print("Analyzing driver performance trends...")
        
        performance_trends = {}
        
        for race_name, race_data in self.races_data.items():
            for driver_id in race_data['Driver'].unique():
                driver_name = self.driver_mapping.get(driver_id, f"Driver_{driver_id}")
                driver_race_data = race_data[race_data['Driver'] == driver_id]
                
                if driver_name not in performance_trends:
                    performance_trends[driver_name] = {
                        'races': [],
                        'avg_lap_times': [],
                        'final_positions': [],
                        'consistency_scores': [],
                        'tyre_management': []
                    }
                
                # Calculate metrics for this race
                avg_lap_time = driver_race_data['LapTime'].mean()
                final_position = driver_race_data['Position'].iloc[-1]
                consistency = 1 / (driver_race_data['LapTime'].std() + 0.001)  # Lower std = higher consistency
                tyre_mgmt = driver_race_data['TyreLife'].max()  # How long tyres lasted
                
                performance_trends[driver_name]['races'].append(race_name)
                performance_trends[driver_name]['avg_lap_times'].append(avg_lap_time)
                performance_trends[driver_name]['final_positions'].append(final_position)
                performance_trends[driver_name]['consistency_scores'].append(consistency)
                performance_trends[driver_name]['tyre_management'].append(tyre_mgmt)
        
        self.performance_trends = performance_trends
        
        print(f"Performance trends calculated for {len(performance_trends)} drivers")
        return performance_trends
    
    def identify_rookie_performance(self):
        """Identify and analyze rookie driver performance"""
        print("Analyzing rookie driver performance...")
        
        # Define 2025 rookies
        rookie_drivers = ['Colapinto', 'Lawson']  # 2025 rookies
        
        rookie_analysis = {}
        
        for rookie in rookie_drivers:
            if rookie in self.performance_trends:
                trends = self.performance_trends[rookie]
                
                # Calculate rookie-specific metrics
                improvement_rate = 0
                if len(trends['final_positions']) > 1:
                    # Calculate position improvement over time
                    positions = trends['final_positions']
                    improvement_rate = (positions[0] - positions[-1]) / len(positions)
                
                rookie_analysis[rookie] = {
                    'races_completed': len(trends['races']),
                    'avg_position': np.mean(trends['final_positions']),
                    'best_position': min(trends['final_positions']),
                    'worst_position': max(trends['final_positions']),
                    'avg_lap_time': np.mean(trends['avg_lap_times']),
                    'consistency': np.mean(trends['consistency_scores']),
                    'improvement_rate': improvement_rate,
                    'championship_points': self.driver_standings[
                        self.driver_standings['Driver'] == rookie
                    ]['points'].iloc[0] if len(self.driver_standings[
                        self.driver_standings['Driver'] == rookie
                    ]) > 0 else 0
                }
        
        self.rookie_analysis = rookie_analysis
        
        print(f"Rookie analysis completed for {len(rookie_analysis)} drivers")
        return rookie_analysis
    
    def generate_performance_report(self):
        """Generate comprehensive performance report"""
        print("Generating comprehensive performance report...")
        
        # Convert DataFrames to JSON-serializable format
        driver_standings_json = self.driver_standings.astype(object).where(pd.notnull(self.driver_standings), None).to_dict('records')
        team_standings_json = self.team_standings.astype(object).where(pd.notnull(self.team_standings), None).to_dict('records')
        
        # Convert numpy types to Python types
        for record in driver_standings_json:
            for key, value in record.items():
                if isinstance(value, (np.integer, np.floating)):
                    record[key] = value.item()
                elif isinstance(value, np.ndarray):
                    record[key] = value.tolist()
        
        for record in team_standings_json:
            for key, value in record.items():
                if isinstance(value, (np.integer, np.floating)):
                    record[key] = value.item()
                elif isinstance(value, np.ndarray):
                    record[key] = value.tolist()
        
        # Convert rookie analysis to JSON-serializable format
        rookie_analysis_json = {}
        for rookie, data in self.rookie_analysis.items():
            rookie_analysis_json[rookie] = {}
            for key, value in data.items():
                if isinstance(value, (np.integer, np.floating)):
                    rookie_analysis_json[rookie][key] = value.item()
                elif isinstance(value, np.ndarray):
                    rookie_analysis_json[rookie][key] = value.tolist()
                else:
                    rookie_analysis_json[rookie][key] = value
        
        report = {
            'season_summary': {
                'year': int(self.year),
                'races_analyzed': int(len(self.races_data)),
                'total_drivers': int(len(self.driver_standings)),
                'total_teams': int(len(self.team_standings)),
                'championship_leader': str(self.driver_standings.iloc[0]['Driver']),
                'constructor_leader': str(self.team_standings.iloc[0]['Team']),
                'most_wins': str(self.driver_standings.loc[
                    self.driver_standings['wins'].idxmax(), 'Driver'
                ]),
                'most_consistent': str(max(
                    self.performance_trends.keys(),
                    key=lambda x: np.mean(self.performance_trends[x]['consistency_scores'])
                )),
                'analysis_date': datetime.now().isoformat()
            },
            'driver_standings': driver_standings_json,
            'team_standings': team_standings_json,
            'rookie_analysis': rookie_analysis_json,
            'performance_insights': self._generate_insights()
        }
        
        # Save report
        report_path = os.path.join(self.output_dir, 'season_performance_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Performance report saved to: {report_path}")
        return report
    
    def _generate_insights(self):
        """Generate key insights from the analysis"""
        insights = []
        
        # Championship battle insight
        top_3 = self.driver_standings.head(3)
        points_gap = top_3.iloc[0]['points'] - top_3.iloc[1]['points']
        insights.append(f"Championship leader {top_3.iloc[0]['Driver']} has a {points_gap} point advantage")
        
        # Rookie performance insight
        if self.rookie_analysis:
            best_rookie = max(
                self.rookie_analysis.keys(),
                key=lambda x: self.rookie_analysis[x]['championship_points']
            )
            insights.append(f"Best performing rookie: {best_rookie} with {self.rookie_analysis[best_rookie]['championship_points']} points")
        
        # Team battle insight
        team_gap = self.team_standings.iloc[0]['points'] - self.team_standings.iloc[1]['points']
        insights.append(f"Constructor's championship gap: {team_gap} points between top 2 teams")
        
        return insights
    
    def create_visualization_data(self):
        """Prepare data for dashboard visualizations"""
        print("Preparing visualization data...")
        
        # Convert driver standings to JSON-serializable format
        driver_viz = self.driver_standings[['Driver', 'points', 'Championship_Position']].to_dict('records')
        team_viz = self.team_standings[['Team', 'points', 'Championship_Position']].to_dict('records')
        
        # Convert numpy types to Python types
        for record in driver_viz:
            for key, value in record.items():
                if isinstance(value, (np.integer, np.floating)):
                    record[key] = value.item()
        
        for record in team_viz:
            for key, value in record.items():
                if isinstance(value, (np.integer, np.floating)):
                    record[key] = value.item()
        
        viz_data = {
            'standings_chart': {
                'drivers': driver_viz,
                'teams': team_viz
            },
            'performance_trends': {},
            'rookie_comparison': {}
        }
        
        # Convert rookie analysis to JSON-serializable format
        for rookie, data in self.rookie_analysis.items():
            viz_data['rookie_comparison'][rookie] = {}
            for key, value in data.items():
                if isinstance(value, (np.integer, np.floating)):
                    viz_data['rookie_comparison'][rookie][key] = value.item()
                elif isinstance(value, np.ndarray):
                    viz_data['rookie_comparison'][rookie][key] = value.tolist()
                else:
                    viz_data['rookie_comparison'][rookie][key] = value
        
        # Prepare trend data for top 10 drivers
        top_drivers = self.driver_standings.head(10)['Driver'].tolist()
        
        for driver in top_drivers:
            if driver in self.performance_trends:
                trends = self.performance_trends[driver]
                viz_data['performance_trends'][driver] = {
                    'races': trends['races'],
                    'positions': [float(pos) if isinstance(pos, (np.integer, np.floating)) else pos for pos in trends['final_positions']],
                    'lap_times': [float(time) if isinstance(time, (np.integer, np.floating)) else time for time in trends['avg_lap_times']]
                }
        
        # Save visualization data
        viz_path = os.path.join(self.output_dir, 'visualization_data.json')
        with open(viz_path, 'w') as f:
            json.dump(viz_data, f, indent=2)
        
        print(f"Visualization data saved to: {viz_path}")
        return viz_data


def main():
    """Main execution function"""
    print("🏎️ F1 Season Analysis Starting...")
    print("=" * 50)
    
    # Initialize analyzer for 2025 season
    analyzer = SeasonStandingsAnalyzer(year=2025)
    
    try:
        # Load race data
        analyzer.load_race_data()
        
        # Calculate standings
        driver_standings = analyzer.calculate_driver_standings()
        team_standings = analyzer.calculate_team_standings()
        
        # Analyze performance trends
        performance_trends = analyzer.analyze_driver_performance_trends()
        
        # Analyze rookies
        rookie_analysis = analyzer.identify_rookie_performance()
        
        # Generate comprehensive report
        report = analyzer.generate_performance_report()
        
        # Prepare visualization data
        viz_data = analyzer.create_visualization_data()
        
        print("\n" + "=" * 50)
        print("✅ Season Analysis Complete!")
        print(f"📊 Championship Leader: {report['season_summary']['championship_leader']}")
        print(f"🏆 Constructor Leader: {report['season_summary']['constructor_leader']}")
        print(f"🆕 Best Rookie: {list(rookie_analysis.keys())[0] if rookie_analysis else 'N/A'}")
        print(f"📈 Races Analyzed: {report['season_summary']['races_analyzed']}")
        
        # Display top 5 standings
        print("\n🏁 Top 5 Driver Standings:")
        for i, row in driver_standings.head(5).iterrows():
            print(f"  {row['Championship_Position']}. {row['Driver']} - {row['points']} points")
        
        print("\n🏭 Top 3 Constructor Standings:")
        for i, row in team_standings.head(3).iterrows():
            print(f"  {row['Championship_Position']}. {row['Team']} - {row['points']} points")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        raise


if __name__ == "__main__":
    main()
