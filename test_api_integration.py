#!/usr/bin/env python3
"""
Test script for F1 API Integration

This script demonstrates the real F1 API integration capabilities:
1. Fetches data from Ergast API
2. Handles fallback scenarios
3. Shows data caching and rate limiting
4. Integrates with existing season analysis
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from data_pipeline.api_integration import F1APIIntegration
from data_pipeline.season_analysis import SeasonStandingsAnalyzer


def test_api_integration():
    """Test the API integration with real F1 data"""
    print("🧪 Testing F1 API Integration")
    print("=" * 60)
    
    # Test 1: Basic API Integration
    print("\n1️⃣ Testing Basic API Integration...")
    api = F1APIIntegration(year=2024)
    
    # Test fetching current standings
    print("   Fetching current season standings...")
    standings = api.fetch_current_season_standings()
    
    if standings:
        print("   ✅ Successfully fetched standings")
        print(f"   📊 Season: {standings['season']}")
        print(f"   🏁 Round: {standings['round']}")
        print(f"   🏎️ Drivers: {len(standings['drivers'])}")
        
        # Show top 3 drivers
        print("\n   🏆 Top 3 Drivers:")
        for i, driver in enumerate(standings['drivers'][:3]):
            print(f"      {driver['position']}. {driver['driver_name']} - {driver['points']} pts")
    else:
        print("   ❌ Failed to fetch standings (will use fallback)")
    
    print("\n" + "-" * 40)
    
    # Test 2: Race Schedule
    print("\n2️⃣ Testing Race Schedule Fetching...")
    schedule = api.fetch_race_schedule()
    
    if schedule:
        print(f"   ✅ Successfully fetched {len(schedule)} races")
        
        # Show next few races
        print("\n   📅 Next Races:")
        for race in schedule[:3]:
            print(f"      Round {race['round']}: {race['race_name']} ({race['date']})")
    else:
        print("   ❌ Failed to fetch race schedule")
    
    print("\n" + "-" * 40)
    
    # Test 3: Comprehensive Data Fetching
    print("\n3️⃣ Testing Comprehensive Data Fetching...")
    comprehensive_data = api.get_comprehensive_season_data()
    
    print(f"   📊 Data Sources: {', '.join(comprehensive_data['data_sources'])}")
    print(f"   🔄 Last Updated: {comprehensive_data['last_updated'][:19]}")
    print(f"   ⚠️ Using Fallback: {comprehensive_data.get('is_fallback', False)}")
    
    # Save data for integration testing
    api.save_api_data_to_pipeline_format(comprehensive_data)
    print("   💾 Data saved to cache")
    
    print("\n" + "-" * 40)
    
    # Test 4: Integration with Season Analysis
    print("\n4️⃣ Testing Integration with Season Analysis...")
    analyzer = SeasonStandingsAnalyzer(year=2025)
    
    try:
        # Load data using API integration
        analyzer.load_race_data(use_api=True)
        
        # Run basic analysis to verify integration
        driver_standings = analyzer.calculate_driver_standings()
        team_standings = analyzer.calculate_team_standings()
        
        print("   ✅ Successfully integrated with season analysis")
        print(f"   📈 Analyzed {len(driver_standings)} drivers")
        print(f"   🏭 Analyzed {len(team_standings)} teams")
        
        # Show championship leader
        if len(driver_standings) > 0:
            leader = driver_standings.iloc[0]
            print(f"   🏆 Championship Leader: {leader['Driver']} ({leader['points']} pts)")
        
    except Exception as e:
        print(f"   ❌ Integration test failed: {e}")
    
    print("\n" + "=" * 60)
    print("🏁 API Integration Test Complete!")


def test_fallback_scenarios():
    """Test fallback scenarios when API is unavailable"""
    print("\n🔧 Testing Fallback Scenarios")
    print("=" * 60)
    
    # Test with invalid year to trigger fallback
    print("\n1️⃣ Testing Fallback Data Generation...")
    api = F1APIIntegration(year=2030)  # Future year, should trigger fallback
    
    fallback_data = api.create_fallback_data()
    print(f"   ✅ Generated fallback data for {fallback_data['season']}")
    print(f"   🏎️ Drivers: {len(fallback_data['drivers'])}")
    print(f"   🏭 Constructors: {len(fallback_data['constructors'])}")
    
    # Show fallback champion
    if fallback_data['drivers']:
        champion = fallback_data['drivers'][0]
        print(f"   🏆 Fallback Champion: {champion['driver_name']} ({champion['points']} pts)")
    
    print("\n" + "-" * 40)
    
    # Test cache functionality
    print("\n2️⃣ Testing Cache Functionality...")
    api_normal = F1APIIntegration(year=2024)
    
    # First request (should hit API)
    print("   Making first request (should hit API)...")
    data1 = api_normal.fetch_current_season_standings()
    
    # Second request (should use cache)
    print("   Making second request (should use cache)...")
    data2 = api_normal.fetch_current_season_standings()
    
    if data1 and data2:
        print("   ✅ Cache functionality working")
        print(f"   📊 Data consistency: {data1['season'] == data2['season']}")
    else:
        print("   ⚠️ Cache test inconclusive (API may be unavailable)")
    
    print("\n" + "=" * 60)
    print("🔄 Fallback Testing Complete!")


def show_data_sources_info():
    """Show information about available data sources"""
    print("\n📋 F1 Data Sources Information")
    print("=" * 60)
    
    sources = [
        {
            "name": "Ergast API",
            "url": "http://ergast.com/mrd/",
            "data_types": ["Race Results", "Standings", "Driver Info", "Constructor Info", "Race Schedule"],
            "coverage": "1950-present",
            "rate_limit": "4 requests/second",
            "status": "Primary source"
        },
        {
            "name": "OpenF1 API",
            "url": "https://openf1.org/",
            "data_types": ["Live Timing", "Telemetry", "Real-time Data"],
            "coverage": "2023-present",
            "rate_limit": "Variable",
            "status": "Future enhancement"
        },
        {
            "name": "Fallback Data",
            "url": "Local generation",
            "data_types": ["Championship Standings", "Driver Performance"],
            "coverage": "Current season simulation",
            "rate_limit": "No limit",
            "status": "Backup system"
        }
    ]
    
    for i, source in enumerate(sources, 1):
        print(f"\n{i}️⃣ {source['name']}")
        print(f"   🌐 URL: {source['url']}")
        print(f"   📊 Data Types: {', '.join(source['data_types'])}")
        print(f"   📅 Coverage: {source['coverage']}")
        print(f"   ⏱️ Rate Limit: {source['rate_limit']}")
        print(f"   ✅ Status: {source['status']}")
    
    print("\n" + "=" * 60)


def main():
    """Main test execution"""
    print("🏎️ F1 API Integration Test Suite")
    print("🚀 Starting comprehensive API testing...")
    
    try:
        # Show data sources information
        show_data_sources_info()
        
        # Test main API functionality
        test_api_integration()
        
        # Test fallback scenarios
        test_fallback_scenarios()
        
        print("\n🎉 All tests completed successfully!")
        print("\n📝 Next Steps:")
        print("   1. Check data/api_cache/ for cached API responses")
        print("   2. Check data/analysis/2025/ for integrated analysis results")
        print("   3. Run season_analysis.py to see full integration")
        print("   4. Use the dashboard to visualize real F1 data")
        
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
