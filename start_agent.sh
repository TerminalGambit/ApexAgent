#!/bin/bash

# F1 ML Dashboard - Agent Launcher
# Usage: ./start_agent.sh [phase_number]

if [ $# -eq 0 ]; then
    echo "🏎️ F1 ML Dashboard - Agent Launcher"
    echo "===================================="
    echo ""
    echo "Usage: ./start_agent.sh [phase_number]"
    echo ""
    echo "Available agents:"
    echo "  1 - Data Pipeline Specialist"
    echo "  2 - Track Analysis Specialist" 
    echo "  3 - Driver Performance Analyst"
    echo "  4 - ML Prediction Specialist"
    echo "  5 - Frontend Integration Specialist"
    echo ""
    echo "Examples:"
    echo "  ./start_agent.sh 1  # Start Data Pipeline Specialist"
    echo "  ./start_agent.sh 2  # Start Track Analysis Specialist"
    echo ""
    exit 1
fi

phase=$1

case $phase in
    1)
        echo "🚀 Starting Data Pipeline Specialist..."
        echo "Workspace: workspace/phase1/"
        echo "Task file: tasks/phase1_data_pipeline.md"
        echo ""
        open workspace/phase1/ 2>/dev/null || echo "Workspace ready at: workspace/phase1/"
        cat tasks/phase1_data_pipeline.md
        ;;
    2)
        echo "🚀 Starting Track Analysis Specialist..."
        echo "Workspace: workspace/phase2/"
        echo "Task file: tasks/phase2_track_analysis.md"
        echo ""
        open workspace/phase2/ 2>/dev/null || echo "Workspace ready at: workspace/phase2/"
        cat tasks/phase2_track_analysis.md
        ;;
    3)
        echo "🚀 Starting Driver Performance Analyst..."
        echo "Workspace: workspace/phase3/"
        echo "Task file: tasks/phase3_rookie_analysis.md"
        echo ""
        open workspace/phase3/ 2>/dev/null || echo "Workspace ready at: workspace/phase3/"
        cat tasks/phase3_rookie_analysis.md
        ;;
    4)
        echo "🚀 Starting ML Prediction Specialist..."
        echo "Workspace: workspace/phase4/"
        echo "Task file: tasks/phase4_silverstone_prediction.md"
        echo ""
        open workspace/phase4/ 2>/dev/null || echo "Workspace ready at: workspace/phase4/"
        cat tasks/phase4_silverstone_prediction.md
        ;;
    5)
        echo "🚀 Starting Frontend Integration Specialist..."
        echo "Workspace: workspace/phase5/"
        echo "Task file: tasks/phase5_dashboard_integration.md"
        echo ""
        open workspace/phase5/ 2>/dev/null || echo "Workspace ready at: workspace/phase5/"
        cat tasks/phase5_dashboard_integration.md
        ;;
    *)
        echo "❌ Invalid phase number: $phase"
        echo "Please use phases 1-5"
        exit 1
        ;;
esac
