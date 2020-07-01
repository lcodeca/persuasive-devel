#!/bin/bash
set -e # error
set -u # unset variables

export RLLIB_SUMO_INFRASTRUCTURE="/home/alice/devel/persuasive-devel"
export DISPLAY=:0

AGENT="$1"
if [ -z "$AGENT" ]
then
    AGENT="agent_0"
fi

# Folders for the graphs
AGGREGATED="graphs/aggregated"
mkdir -p $AGGREGATED
AGENTS="graphs/agents"
mkdir -p $AGENTS
ETT="graphs/ett"
mkdir -p $ETT
QVALUES="graphs/qvalues"
mkdir -p $QVALUES
POLICIES="graphs/policies"
mkdir -p $POLICIES
QVALUESEVOL="graphs/qvalues-evol/$AGENT"
mkdir -p $QVALUESEVOL

# Cleanup
rm -rf $QVALUES/*
rm -rf $POLICIES/*
rm -rf $QVALUESEVOL/*

# Results directory
RESULTS="res/logs/results"

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

python3 $RLLIB_SUMO_INFRASTRUCTURE/src/utils/DBLoggerCustomStats/ett.insight.py \
    --dir-tree $RESULTS \
    --graph "$ETT/ett" \
    --last-run

python3 $RLLIB_SUMO_INFRASTRUCTURE/src/utils/DBLoggerCustomStats/policy.insight.py \
    --dir-tree $RESULTS --graph "$POLICIES/policy" --last-run

python3 $RLLIB_SUMO_INFRASTRUCTURE/src/utils/DBLoggerCustomStats/qvalues.insight.py \
    --dir-tree $RESULTS --graph "$QVALUES/qvalues" --last-run

python3 $RLLIB_SUMO_INFRASTRUCTURE/src/utils/DBLoggerCustomStats/qvalues.insight.py \
    --dir-tree $RESULTS --graph "$QVALUESEVOL/qval.evol" --agent "$AGENT"
