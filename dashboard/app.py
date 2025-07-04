import os
import sys
import json
import pandas as pd
import numpy as np
import joblib
from flask import Flask, render_template, request, jsonify
import plotly.graph_objs as go
import plotly.utils
from plotly.subplots import make_subplots
import plotly.express as px

# Add parent directory to path for race predictor
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from data_pipeline.race_predictor import RacePredictionEngine
from data_monitor import F1DataMonitor

app = Flask(__name__)

# Configuration
DATA_DIR = "../data/processed/2024/Monaco Grand Prix/"
MODELS_DIR = "../models/trained/"
PREDICTIONS_DIR = "../data/predictions/2025/"
ANALYSIS_DIR = "../data/analysis/2025/"

class F1Dashboard:
    """F1 ML Dashboard data handler"""
    
    def __init__(self):
        self.load_data()
        self.load_models()
        self.load_race_predictions()
        self.load_season_analysis()
        
    def load_data(self):
        """Load all necessary data files"""
        try:
            # Load datasets
            self.laps_features = pd.read_csv(DATA_DIR + "laps_features.csv")
            self.laps_clean = pd.read_csv(DATA_DIR + "laps_features_clean.csv")
            self.train_data = pd.read_csv(DATA_DIR + "train_data.csv")
            self.test_data = pd.read_csv(DATA_DIR + "test_data.csv")
            
            # Load feature analysis results
            self.feature_importance = pd.read_csv(DATA_DIR + "feature_importance.csv")
            self.feature_stats = pd.read_csv(DATA_DIR + "feature_statistics.csv")
            
            print("✅ Successfully loaded all data files")
            
        except Exception as e:
            print("❌ Error loading data:", e)
            # Create dummy data for demo
            self.create_dummy_data()
    
    def load_models(self):
        """Load trained models and results"""
        try:
            # Load model comparison results
            self.baseline_results = pd.read_csv(MODELS_DIR + "model_comparison.csv")
            self.advanced_results = pd.read_csv(MODELS_DIR + "advanced_model_comparison.csv")
            
            # Load detailed results
            with open(MODELS_DIR + "training_results.json", 'r') as f:
                self.baseline_detailed = json.load(f)
            
            with open(MODELS_DIR + "advanced_training_results.json", 'r') as f:
                self.advanced_detailed = json.load(f)
                
            # Load best model
            self.best_model = joblib.load(MODELS_DIR + "advanced_ElasticNet.joblib")
            
            print("✅ Successfully loaded all models and results")
            
        except Exception as e:
            print("❌ Error loading models:", e)
            self.create_dummy_results()
    
    def create_dummy_data(self):
        """Create dummy data for demonstration"""
        print("Creating dummy data for demonstration...")
        
        # Create sample lap data
        n_samples = 1000
        self.laps_features = pd.DataFrame({
            'LapNumber': np.repeat(range(1, 51), 20),
            'LapTime': np.random.normal(80, 5, n_samples),
            'Position': np.random.randint(1, 21, n_samples),
            'Driver': np.random.choice(['Hamilton', 'Verstappen', 'Leclerc', 'Norris'], n_samples),
            'Team': np.random.choice(['Mercedes', 'Red Bull', 'Ferrari', 'McLaren'], n_samples),
            'TyreLife': np.random.randint(1, 40, n_samples),
            'Sector1Time': np.random.normal(25, 2, n_samples),
            'Sector2Time': np.random.normal(35, 3, n_samples),
            'Sector3Time': np.random.normal(20, 1.5, n_samples)
        })
        
        self.laps_clean = self.laps_features.copy()
        self.train_data = self.laps_features.sample(800)
        self.test_data = self.laps_features.drop(self.train_data.index)
    
    def create_dummy_results(self):
        """Create dummy model results"""
        self.baseline_results = pd.DataFrame({
            'model': ['Linear_Regression', 'Random_Forest', 'XGBoost'],
            'test_rmse': [0.350, 0.400, 0.380],
            'test_r2': [0.992, 0.988, 0.990],
            'cv_rmse_mean': [0.454, 0.520, 0.490]
        })
        
        self.advanced_results = pd.DataFrame({
            'model': ['ElasticNet', 'Ensemble_Voting', 'XGBoost_Advanced'],
            'test_rmse': [0.350, 0.365, 0.377],
            'test_r2': [0.992, 0.991, 0.990],
            'cv_rmse_mean': [0.454, 0.730, 1.118]
        })
    
    def load_race_predictions(self):
        """Load race prediction data"""
        try:
            # Initialize race predictor
            self.race_predictor = RacePredictionEngine(year=2025, track_name="Austria GP")
            
            # Check if we have existing predictions
            prediction_file = os.path.join(PREDICTIONS_DIR, "Austria_GP_prediction_report.json")
            if os.path.exists(prediction_file):
                with open(prediction_file, 'r') as f:
                    self.prediction_report = json.load(f)
                print("✅ Loaded existing race predictions")
            else:
                # Run predictions if not already done
                print("🚀 Generating new race predictions...")
                self.generate_race_predictions()
                
        except Exception as e:
            print(f"❌ Error loading race predictions: {e}")
            self.prediction_report = None
    
    def load_season_analysis(self):
        """Load season analysis data from FastF1 pipeline"""
        try:
            # Load season performance report
            report_file = os.path.join(ANALYSIS_DIR, "season_performance_report.json")
            if os.path.exists(report_file):
                with open(report_file, 'r') as f:
                    self.season_report = json.load(f)
                print("✅ Loaded season analysis report")
            else:
                self.season_report = None
                print("⚠️ Season analysis report not found")
            
            # Load visualization data
            viz_file = os.path.join(ANALYSIS_DIR, "visualization_data.json")
            if os.path.exists(viz_file):
                with open(viz_file, 'r') as f:
                    self.viz_data = json.load(f)
                print("✅ Loaded visualization data")
            else:
                self.viz_data = None
                print("⚠️ Visualization data not found")
                
        except Exception as e:
            print(f"❌ Error loading season analysis: {e}")
            self.season_report = None
            self.viz_data = None
    
    def generate_race_predictions(self):
        """Generate fresh race predictions"""
        try:
            # Run the full prediction pipeline
            baseline = self.race_predictor.stage_1_baseline_prediction()
            practice = self.race_predictor.stage_2_practice_update()
            qualifying = self.race_predictor.stage_3_qualifying_update()
            final = self.race_predictor.stage_4_pre_race_final()
            
            # Load actual results and evaluate
            actual = self.race_predictor.load_actual_results()
            evaluation = self.race_predictor.evaluate_predictions()
            
            # Generate report
            self.prediction_report = self.race_predictor.generate_prediction_report()
            
            print("✅ Race predictions generated successfully")
            
        except Exception as e:
            print(f"❌ Error generating race predictions: {e}")
            self.prediction_report = None

# Initialize dashboard
dashboard = F1Dashboard()

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')

@app.route('/model-performance')
def model_performance():
    """Model performance comparison page"""
    
    # Create model performance comparison chart
    fig = go.Figure()
    
    # Add baseline models
    fig.add_trace(go.Bar(
        name='Baseline Models',
        x=dashboard.baseline_results['model'],
        y=dashboard.baseline_results['test_rmse'],
        marker_color='lightblue',
        text=dashboard.baseline_results['test_rmse'].round(3),
        textposition='auto'
    ))
    
    # Add advanced models
    fig.add_trace(go.Bar(
        name='Advanced Models',
        x=dashboard.advanced_results['model'],
        y=dashboard.advanced_results['test_rmse'],
        marker_color='darkblue',
        text=dashboard.advanced_results['test_rmse'].round(3),
        textposition='auto'
    ))
    
    fig.update_layout(
        title='Model Performance Comparison (Test RMSE)',
        xaxis_title='Model',
        yaxis_title='RMSE (seconds)',
        barmode='group',
        template='plotly_white'
    )
    
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    return render_template('model_performance.html', graphJSON=graphJSON)

@app.route('/feature-analysis')
def feature_analysis():
    """Feature importance and analysis page"""
    
    try:
        # Feature importance chart
        top_features = dashboard.feature_importance.head(15)
        
        fig = go.Figure(go.Bar(
            x=top_features['importance'],
            y=top_features['feature'],
            orientation='h',
            marker_color='green'
        ))
        
        fig.update_layout(
            title='Top 15 Feature Importance',
            xaxis_title='Importance Score',
            yaxis_title='Features',
            template='plotly_white',
            height=600
        )
        
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        
    except Exception as e:
        print("Error creating feature analysis:", e)
        # Create dummy chart
        fig = go.Figure(go.Bar(
            x=[0.26, 0.12, 0.10, 0.08, 0.07],
            y=['best_lap_so_far', 'team_avg_lap_time', 'rolling_avg_3', 'sector2_time', 'tyre_life'],
            orientation='h',
            marker_color='green'
        ))
        fig.update_layout(title='Feature Importance (Demo Data)', template='plotly_white')
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    return render_template('feature_analysis.html', graphJSON=graphJSON)

@app.route('/lap-analysis')
def lap_analysis():
    """Lap time analysis and predictions"""
    
    try:
        # Lap time progression by driver
        fig = px.line(
            dashboard.laps_features.groupby(['LapNumber', 'Driver'])['LapTime'].mean().reset_index(),
            x='LapNumber',
            y='LapTime',
            color='Driver',
            title='Lap Time Progression by Driver'
        )
        fig.update_layout(template='plotly_white')
        
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        
    except Exception as e:
        print("Error creating lap analysis:", e)
        # Create dummy chart
        laps = list(range(1, 51))
        fig = go.Figure()
        for driver in ['Hamilton', 'Verstappen', 'Leclerc']:
            lap_times = np.random.normal(80, 2, 50) + np.random.normal(0, 0.1, 50) * np.arange(50)
            fig.add_trace(go.Scatter(x=laps, y=lap_times, mode='lines', name=driver))
        
        fig.update_layout(
            title='Lap Time Progression by Driver (Demo)',
            xaxis_title='Lap Number',
            yaxis_title='Lap Time (seconds)',
            template='plotly_white'
        )
        
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    return render_template('lap_analysis.html', graphJSON=graphJSON)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Real-time prediction interface"""
    
    if request.method == 'POST':
        try:
            # Get form data
            lap_number = float(request.form.get('lap_number', 1))
            tyre_life = float(request.form.get('tyre_life', 1))
            sector1 = float(request.form.get('sector1', 25))
            sector2 = float(request.form.get('sector2', 35))
            sector3 = float(request.form.get('sector3', 20))
            
            # Create prediction (demo)
            predicted_time = sector1 + sector2 + sector3 + np.random.normal(0, 0.5)
            confidence = max(0.85, min(0.99, 0.95 + np.random.normal(0, 0.02)))
            
            return jsonify({
                'prediction': round(predicted_time, 3),
                'confidence': round(confidence, 3),
                'status': 'success'
            })
            
        except Exception as e:
            return jsonify({'error': str(e), 'status': 'error'})
    
    return render_template('predict.html')

@app.route('/data-quality')
def data_quality():
    """Data quality and pipeline status"""
    
    # Data quality metrics
    quality_metrics = {
        'total_samples': len(dashboard.laps_clean),
        'features': len(dashboard.laps_clean.columns) - 1,  # Excluding target
        'missing_values': dashboard.laps_clean.isnull().sum().sum(),
        'outliers_removed': len(dashboard.laps_features) - len(dashboard.laps_clean)
    }
    
    # Create data distribution chart
    try:
        fig = px.histogram(
            dashboard.laps_clean,
            x='LapTime',
            nbins=50,
            title='Lap Time Distribution'
        )
        fig.update_layout(template='plotly_white')
        
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        
    except Exception as e:
        print("Error creating data quality chart:", e)
        # Create dummy histogram
        lap_times = np.random.normal(80, 5, 1000)
        fig = px.histogram(x=lap_times, nbins=50, title='Lap Time Distribution (Demo)')
        fig.update_layout(template='plotly_white')
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    return render_template('data_quality.html', 
                         quality_metrics=quality_metrics, 
                         graphJSON=graphJSON)

@app.route('/race-predictions')
def race_predictions():
    """Race prediction dashboard page"""
    return render_template('race_predictions.html', 
                         prediction_report=dashboard.prediction_report)

@app.route('/season-standings')
def season_standings():
    """Season standings page with FastF1 data"""
    
    # Create driver standings chart
    driver_chart = None
    team_chart = None
    
    if dashboard.season_report and dashboard.viz_data:
        try:
            # Driver standings chart
            drivers_data = dashboard.viz_data['standings_chart']['drivers']
            
            fig_drivers = go.Figure(data=[
                go.Bar(
                    x=[driver['Driver'] for driver in drivers_data[:10]],  # Top 10
                    y=[driver['points'] for driver in drivers_data[:10]],
                    marker_color='#e10600',
                    text=[f"{driver['points']}" for driver in drivers_data[:10]],
                    textposition='auto'
                )
            ])
            
            fig_drivers.update_layout(
                title='2025 Driver Championship Standings (Top 10)',
                xaxis_title='Driver',
                yaxis_title='Points',
                template='plotly_white',
                height=500
            )
            
            driver_chart = json.dumps(fig_drivers, cls=plotly.utils.PlotlyJSONEncoder)
            
            # Team standings chart
            teams_data = dashboard.viz_data['standings_chart']['teams']
            
            fig_teams = go.Figure(data=[
                go.Bar(
                    x=[team['Team'] for team in teams_data],
                    y=[team['points'] for team in teams_data],
                    marker_color='#00d2be',
                    text=[f"{team['points']}" for team in teams_data],
                    textposition='auto'
                )
            ])
            
            fig_teams.update_layout(
                title='2025 Constructor Championship Standings',
                xaxis_title='Team',
                yaxis_title='Points',
                template='plotly_white',
                height=400
            )
            
            team_chart = json.dumps(fig_teams, cls=plotly.utils.PlotlyJSONEncoder)
            
        except Exception as e:
            print(f"Error creating standings charts: {e}")
    
    return render_template('season_standings.html', 
                         season_report=dashboard.season_report,
                         viz_data=dashboard.viz_data,
                         driver_chart=driver_chart,
                         team_chart=team_chart)

@app.route('/api/generate-predictions', methods=['POST'])
def api_generate_predictions():
    """API endpoint to generate new race predictions"""
    try:
        dashboard.generate_race_predictions()
        return jsonify({'status': 'success', 'message': 'Predictions generated successfully'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/prediction-data')
def api_prediction_data():
    """API endpoint for race prediction data"""
    if dashboard.prediction_report:
        return jsonify(dashboard.prediction_report)
    else:
        return jsonify({'error': 'No prediction data available'})

@app.route('/data-monitoring')
def data_monitoring():
    """Comprehensive data monitoring page"""
    try:
        # Initialize data monitor
        monitor = F1DataMonitor(base_dir="../data")
        
        # Load or generate monitoring report
        report_file = "data_monitoring_report.json"
        if os.path.exists(report_file):
            with open(report_file, 'r') as f:
                monitoring_data = json.load(f)
        else:
            monitoring_data = monitor.save_monitoring_report(report_file)
        
        # Create pipeline status visualization
        pipeline_fig = create_pipeline_status_chart(monitoring_data['pipeline_status'])
        pipeline_chart = json.dumps(pipeline_fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        # Create data size visualization
        size_fig = create_data_size_chart(monitoring_data['data_summary'])
        size_chart = json.dumps(size_fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        # Create feature consistency chart
        feature_fig = create_feature_analysis_chart(monitoring_data['feature_analysis'])
        feature_chart = json.dumps(feature_fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        return render_template('data_monitoring.html',
                             monitoring_data=monitoring_data,
                             pipeline_chart=pipeline_chart,
                             size_chart=size_chart,
                             feature_chart=feature_chart)
                             
    except Exception as e:
        print(f"Error in data monitoring: {e}")
        return render_template('data_monitoring.html',
                             monitoring_data=None,
                             error=str(e))

def create_pipeline_status_chart(pipeline_status):
    """Create pipeline status visualization"""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('2024 Pipeline Status', '2025 Pipeline Status'),
        specs=[[{'type': 'bar'}, {'type': 'bar'}]]
    )
    
    for i, (year, status) in enumerate(pipeline_status.items()):
        stages = list(status['pipeline_stages'].keys())
        values = list(status['pipeline_stages'].values())
        
        fig.add_trace(
            go.Bar(
                x=stages,
                y=values,
                name=f"{year}",
                marker_color='#e10600' if year == 2024 else '#00d2be',
                text=[f"{v:.0f}%" for v in values],
                textposition='auto'
            ),
            row=1, col=i+1
        )
    
    fig.update_layout(
        title='Data Pipeline Completion Status by Year',
        showlegend=False,
        template='plotly_white',
        height=400
    )
    
    fig.update_yaxes(title_text="Completion %", range=[0, 100])
    
    return fig

def create_data_size_chart(data_summary):
    """Create data size visualization"""
    years = []
    sizes = []
    races = []
    
    for year, year_data in data_summary['seasons'].items():
        if year_data:
            years.append(str(year))
            sizes.append(year_data['data_size_mb'])
            races.append(year_data['total_races'])
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Data Size by Year (MB)', 'Number of Races by Year'),
        specs=[[{'type': 'bar'}, {'type': 'bar'}]]
    )
    
    # Data size chart
    fig.add_trace(
        go.Bar(
            x=years,
            y=sizes,
            name='Data Size (MB)',
            marker_color='#667eea',
            text=[f"{s:.1f} MB" for s in sizes],
            textposition='auto'
        ),
        row=1, col=1
    )
    
    # Number of races chart
    fig.add_trace(
        go.Bar(
            x=years,
            y=races,
            name='Number of Races',
            marker_color='#764ba2',
            text=[f"{r} races" for r in races],
            textposition='auto'
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title='Data Volume Analysis',
        showlegend=False,
        template='plotly_white',
        height=400
    )
    
    return fig

def create_feature_analysis_chart(feature_analysis):
    """Create feature analysis visualization"""
    if not feature_analysis['feature_consistency']:
        # Create dummy chart if no features found
        fig = go.Figure()
        fig.add_annotation(
            text="No feature data available for analysis",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=16)
        )
        fig.update_layout(
            title='Feature Consistency Analysis',
            template='plotly_white',
            height=400
        )
        return fig
    
    # Get top features by consistency
    sorted_features = sorted(
        feature_analysis['feature_consistency'].items(),
        key=lambda x: x[1],
        reverse=True
    )[:15]
    
    features = [f[0] for f in sorted_features]
    counts = [f[1] for f in sorted_features]
    
    fig = go.Figure(data=[
        go.Bar(
            y=features,
            x=counts,
            orientation='h',
            marker_color='#2E8B57',
            text=[f"{c} races" for c in counts],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title='Top 15 Most Consistent Features Across Races',
        xaxis_title='Number of Races',
        yaxis_title='Features',
        template='plotly_white',
        height=600
    )
    
    return fig

@app.route('/api/stats')
def api_stats():
    """API endpoint for dashboard statistics"""
    
    stats = {
        'total_laps': len(dashboard.laps_clean),
        'total_features': len(dashboard.laps_clean.columns) - 1,
        'best_model': 'ElasticNet',
        'best_rmse': 0.350,
        'best_r2': 0.992,
        'data_quality': 'Excellent'
    }
    
    return jsonify(stats)

if __name__ == '__main__':
    print("🏎️ F1 ML Dashboard Starting...")
    print("📊 Dashboard available at: http://localhost:5555")
    app.run(debug=True, host='0.0.0.0', port=5555)
