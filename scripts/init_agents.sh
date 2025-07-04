#!/bin/bash

# F1 ML Dashboard - Agent Initialization Script
# This script sets up and initializes specialized agents for parallel development

echo "🏎️ F1 ML Dashboard - Agent Initialization"
echo "=========================================="

# Create necessary directories
mkdir -p logs/agents
mkdir -p workspace/phase{1,2,3,4,5}

# Function to start an agent
start_agent() {
    local phase=$1
    local agent_name=$2
    local task_file=$3
    
    echo "🚀 Starting $agent_name (Phase $phase)..."
    
    # Create workspace for this agent
    mkdir -p "workspace/phase$phase"
    
    # Copy task file to workspace
    cp "$task_file" "workspace/phase$phase/"
    
    # Create agent log file
    agent_log=$(echo "${agent_name}" | tr '[:upper:]' '[:lower:]' | tr '_' '-')
    touch "logs/agents/phase${phase}_${agent_log}.log"
    
    echo "Agent: $agent_name" > "workspace/phase$phase/agent_info.txt"
    echo "Phase: $phase" >> "workspace/phase$phase/agent_info.txt"
    echo "Task File: $task_file" >> "workspace/phase$phase/agent_info.txt"
    echo "Workspace: workspace/phase$phase/" >> "workspace/phase$phase/agent_info.txt"
    echo "Log File: logs/agents/phase${phase}_${agent_log}.log" >> "workspace/phase$phase/agent_info.txt"
    echo "Started: $(date)" >> "workspace/phase$phase/agent_info.txt"
    
    echo "✅ $agent_name workspace ready at: workspace/phase$phase/"
}

# Initialize all agents
echo ""
echo "Initializing specialized agents..."
echo ""

start_agent "1" "Data_Pipeline_Specialist" "tasks/phase1_data_pipeline.md"
start_agent "2" "Track_Analysis_Specialist" "tasks/phase2_track_analysis.md" 
start_agent "3" "Driver_Performance_Analyst" "tasks/phase3_rookie_analysis.md"
start_agent "4" "ML_Prediction_Specialist" "tasks/phase4_silverstone_prediction.md"
start_agent "5" "Frontend_Integration_Specialist" "tasks/phase5_dashboard_integration.md"

echo ""
echo "🎯 All agents initialized successfully!"
echo ""
echo "Next steps:"
echo "1. Run individual agent commands below"
echo "2. Each agent will work in their specialized workspace"
echo "3. Monitor progress with: tail -f logs/agents/phase*_*.log"
echo ""
echo "=========================================="
