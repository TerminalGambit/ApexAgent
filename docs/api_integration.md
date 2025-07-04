# F1 API Integration Documentation

## Overview

The F1 API Integration system provides real-time access to Formula 1 data from multiple sources, with built-in error handling, caching, and fallback mechanisms. This ensures your F1-ML pipeline always has access to current season data.

## Features

### 🌐 Multiple Data Sources
- **Ergast API**: Historical and current season data (1950-present)
- **OpenF1 API**: Live timing and telemetry (planned enhancement)
- **Fallback System**: Local data generation when APIs are unavailable

### 🚀 Performance & Reliability
- **Intelligent Caching**: Reduces API calls and improves response times
- **Rate Limiting**: Respects API rate limits automatically
- **Error Handling**: Graceful degradation with fallback data
- **Connection Pooling**: Efficient HTTP session management

### 📊 Data Types Available
- Championship standings (drivers & constructors)
- Race results and qualifying results
- Race schedules and circuit information
- Driver and constructor profiles
- Historical performance data

## Quick Start

### Basic Usage

```python
from data_pipeline.api_integration import F1APIIntegration

# Initialize for current season
api = F1APIIntegration(year=2024)

# Fetch comprehensive season data
season_data = api.get_comprehensive_season_data()

# Check data sources used
print(f"Data from: {', '.join(season_data['data_sources'])}")

# Save for pipeline integration
api.save_api_data_to_pipeline_format(season_data)
```

### Integration with Season Analysis

```python
from data_pipeline.season_analysis import SeasonStandingsAnalyzer

# Initialize analyzer with API integration
analyzer = SeasonStandingsAnalyzer(year=2025)

# Load data using API (with fallback)
analyzer.load_race_data(use_api=True)

# Continue with normal analysis
standings = analyzer.calculate_driver_standings()
```

## API Classes and Methods

### F1APIIntegration Class

#### Constructor
```python
F1APIIntegration(year=2024, cache_dir="data/cache")
```
- `year`: Championship year to fetch data for
- `cache_dir`: Directory for cached API responses

#### Core Methods

##### `fetch_current_season_standings()`
Retrieves driver and constructor championship standings.

**Returns:**
```python
{
    'drivers': [
        {
            'position': 1,
            'points': 575,
            'wins': 19,
            'driver_name': 'Max Verstappen',
            'team_name': 'Red Bull Racing Honda RBPT'
        },
        # ... more drivers
    ],
    'constructors': [
        {
            'position': 1,
            'points': 727,
            'wins': 19,
            'team_name': 'Red Bull Racing Honda RBPT'
        },
        # ... more teams
    ],
    'season': 2024,
    'round': 22
}
```

##### `fetch_race_schedule()`
Gets the complete race calendar for the season.

**Returns:**
```python
[
    {
        'round': 1,
        'race_name': 'Bahrain Grand Prix',
        'circuit_name': 'Bahrain International Circuit',
        'location': 'Sakhir, Bahrain',
        'date': '2024-03-02',
        'coordinates': {'lat': 26.0325, 'lng': 50.5106}
    },
    # ... more races
]
```

##### `fetch_race_results(round_number=None)`
Retrieves race results for specific rounds or all races.

##### `fetch_qualifying_results(round_number=None)`
Gets qualifying session results.

##### `get_comprehensive_season_data()`
Fetches all available data from multiple sources.

**Returns:**
```python
{
    'year': 2024,
    'last_updated': '2024-12-30T21:35:20',
    'data_sources': ['ergast_standings', 'ergast_schedule', 'ergast_results'],
    'standings': {...},
    'schedule': [...],
    'race_results': {...},
    'is_fallback': False
}
```

## Data Sources

### 1. Ergast API
- **URL**: http://ergast.com/mrd/
- **Coverage**: 1950-present
- **Rate Limit**: ~4 requests/second
- **Data Quality**: High, official F1 data
- **Availability**: Very reliable

**Example Endpoints:**
- Standings: `http://ergast.com/api/f1/2024/driverStandings.json`
- Results: `http://ergast.com/api/f1/2024/results.json`
- Schedule: `http://ergast.com/api/f1/2024.json`

### 2. OpenF1 API (Future Enhancement)
- **URL**: https://openf1.org/
- **Coverage**: 2023-present
- **Data Type**: Live timing, telemetry
- **Status**: Planned for future releases

### 3. Fallback System
When APIs are unavailable, the system generates realistic data based on:
- Historical performance patterns
- Current season context
- Driver/team characteristics

## Caching System

### Cache Structure
```
data/cache/
├── standings_2024.json
├── schedule_2024.json
├── results_2024_all.json
└── drivers_2024.json
```

### Cache Behavior
- **Standings**: 2-hour cache (dynamic data)
- **Schedule**: 24-hour cache (static data)
- **Results**: 1-hour cache (race weekends)
- **Driver Info**: 24-hour cache (rarely changes)

### Cache Management
```python
# Clear cache manually
import shutil
shutil.rmtree("data/cache")

# Check cache status
cache_path = Path("data/cache")
print(f"Cache size: {sum(f.stat().st_size for f in cache_path.rglob('*') if f.is_file())} bytes")
```

## Error Handling

### Automatic Fallbacks
1. **Network Issues**: Retry with exponential backoff
2. **API Unavailable**: Switch to fallback data generation
3. **Rate Limiting**: Automatic rate limiting with delays
4. **Data Corruption**: Validation and re-fetching

### Error Scenarios
```python
# Handle specific error cases
try:
    data = api.get_comprehensive_season_data()
    if data.get('is_fallback'):
        print("⚠️ Using fallback data - API unavailable")
except Exception as e:
    print(f"❌ API integration failed: {e}")
    # Application continues with fallback data
```

## Performance Optimization

### Best Practices
1. **Use Caching**: Don't disable cache unless necessary
2. **Batch Requests**: Use comprehensive data fetching
3. **Rate Limiting**: Let the system handle rate limiting
4. **Error Recovery**: Always handle fallback scenarios

### Monitoring
```python
# Check API performance
season_data = api.get_comprehensive_season_data()
print(f"Sources used: {season_data['data_sources']}")
print(f"Fallback mode: {season_data.get('is_fallback', False)}")
```

## Integration Examples

### Example 1: Real-time Dashboard Data
```python
def update_dashboard():
    api = F1APIIntegration(year=2024)
    
    # Get latest standings
    standings = api.fetch_current_season_standings()
    
    if standings and not standings.get('is_fallback'):
        # Update dashboard with real data
        update_live_standings(standings)
    else:
        # Show cached or fallback data
        show_fallback_message()
```

### Example 2: Race Weekend Updates
```python
def process_race_weekend(round_number):
    api = F1APIIntegration(year=2024)
    
    # Get race results
    results = api.fetch_race_results(round_number)
    qualifying = api.fetch_qualifying_results(round_number)
    
    # Process and analyze
    if results and qualifying:
        analyze_race_performance(results, qualifying)
```

### Example 3: Historical Analysis
```python
def analyze_season_trends():
    api = F1APIIntegration(year=2024)
    
    # Get complete season data
    season_data = api.get_comprehensive_season_data()
    
    # Save for ML training
    api.save_api_data_to_pipeline_format(season_data)
    
    # Continue with analysis
    return season_data
```

## Testing

### Run Test Suite
```bash
# Test all API functionality
python test_api_integration.py

# Test specific components
python -c "from data_pipeline.api_integration import F1APIIntegration; api = F1APIIntegration(); print(api.fetch_current_season_standings())"
```

### Test Scenarios
1. **Live API Test**: Fetch real data from Ergast
2. **Fallback Test**: Simulate API failure
3. **Cache Test**: Verify caching behavior
4. **Integration Test**: Full pipeline integration

## Troubleshooting

### Common Issues

#### 1. API Connection Timeout
```python
# Increase timeout
api = F1APIIntegration(year=2024)
data = api._make_request(url, "ergast", timeout=30)
```

#### 2. Rate Limiting
```python
# Check rate limiting status
print(f"Last request times: {api.last_request_time}")
```

#### 3. Cache Issues
```python
# Clear and rebuild cache
shutil.rmtree("data/cache")
api = F1APIIntegration(year=2024)
data = api.get_comprehensive_season_data()
```

#### 4. Integration Problems
```python
# Check data format compatibility
analyzer = SeasonStandingsAnalyzer(year=2025)
try:
    analyzer.load_race_data(use_api=True)
except Exception as e:
    print(f"Integration issue: {e}")
    # Fall back to local data
    analyzer.load_race_data(use_api=False)
```

## Future Enhancements

### Planned Features
1. **OpenF1 Integration**: Live timing and telemetry
2. **WebSocket Support**: Real-time updates during races
3. **Advanced Caching**: Redis/database support
4. **API Key Management**: Support for premium APIs
5. **Data Validation**: Enhanced data quality checks

### Contributing
To contribute to API integration:
1. Add new data sources in `api_integration.py`
2. Implement new endpoint methods
3. Add comprehensive tests
4. Update documentation

## API Limits and Fair Usage

### Ergast API Guidelines
- Maximum 4 requests per second
- No API key required
- Free for non-commercial use
- Respect server resources

### Best Practices
- Use caching to minimize requests
- Implement exponential backoff
- Monitor your usage patterns
- Consider fallback scenarios

## Support

For issues with API integration:
1. Check the test suite: `python test_api_integration.py`
2. Review logs in the console output
3. Verify network connectivity
4. Check API status at source websites

## License and Attribution

This integration uses:
- **Ergast API**: © Ergast Ltd. - Used under fair use
- **OpenF1 API**: © OpenF1 - Planned integration
- **Project Code**: MIT License

Always respect API terms of service and attribution requirements.
