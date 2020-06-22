#!/bin/bash

export RLLIB_SUMO_INFRASTRUCTURE="/home/alice/devel/persuasive-devel"

EXP=$1
AGENT="$2"
if [ -z "$AGENT" ]
then
    AGENT="agent_0"
fi

# Folders for the graphs
AGGREGATED="$EXP/graphs/aggregated"
mkdir -p $AGGREGATED
AGENTS="$EXP/graphs/agents"
mkdir -p $AGENTS
ETT="$EXP/graphs/ett"
mkdir -p $ETT

# Results directory
RESULTS="$EXP/logs/results"

python3 $RLLIB_SUMO_INFRASTRUCTURE/src/utils/DBLoggerCustomStats/aggregated.overview.py \
    --dir-tree $RESULTS --window 5 \
    --data "$AGGREGATED/aggregated-overview.json" \
    --graph "$AGGREGATED/aggregated-overview" # --profiler

python3 $RLLIB_SUMO_INFRASTRUCTURE/src/utils/DBLoggerCustomStats/mode.share.over.episodes.py \
    --dir-tree $RESULTS \
    --data "$AGGREGATED/mode-share-overview.json" \
    --graph "$AGGREGATED/mode-share-overview" # --profiler

python3 $RLLIB_SUMO_INFRASTRUCTURE/src/utils/DBLoggerCustomStats/aggregated.agents.decisions.py \
    --dir-tree $RESULTS \
    --data "$AGGREGATED/agents-decisions-overview.json" \
    --graph "$AGGREGATED/agents-decisions-overview" # --profiler

python3 $RLLIB_SUMO_INFRASTRUCTURE/src/utils/DBLoggerCustomStats/agent.overview.py \
    --dir-tree $RESULTS \
    --data "$AGENTS/agents-overview.json" \
    --graph "$AGENTS/overview" # --profiler
