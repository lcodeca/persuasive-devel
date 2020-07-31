#!/bin/bash
set -e
set -u

export DISPLAY=:0

export SUMO_HOME="/home/alice/sumo"
export PATH="$SUMO_HOME/bin:$PATH"

export RLLIB_SUMO_INFRASTRUCTURE="/home/alice/devel/persuasive-devel"

# Folder for the sumo simulation
SIMULATION="sumo"
mkdir -p $SIMULATION

AGENT="agent_0"

# Folders for the graphs
GRAPHS="res_PDEGQLET_MARLCoop/graphs"
AGGREGATED="$GRAPHS/aggregated"
mkdir -p $AGGREGATED
AGENTS="$GRAPHS/agents"
mkdir -p $AGENTS
ETT="$GRAPHS/ett"
mkdir -p $ETT
QVALUES="$GRAPHS/qvalues"
mkdir -p $QVALUES
POLICIES="$GRAPHS/policies"
mkdir -p $POLICIES
QVALUESEVOL="$GRAPHS/qvalues-evol/$AGENT"
mkdir -p $QVALUESEVOL

echo "TEST for QLSA Training Stats" > QLSA-test-training-stats.log

echo "[$(date)] RUNNING PDEGQLET - MARLCoop" | tee -a QLSA-test-training-stats.log
for i in {1..5} ;do
    echo "[$(date)] Iteration $i " | tee -a QLSA-test-training-stats.log
    python3 -u $RLLIB_SUMO_INFRASTRUCTURE/src/runQLSAtraining.py \
        --algo PDEGQLET --env MARLCoop \
        --epsilon 0.1 --action-distr 0.9 0.1 \
        --dir "res_PDEGQLET_MARLCoop" --config "configuration.json" \
        --checkout-steps 3 --training-steps 1 \
        --alpha 0.1 --gamma 0.9 --decay 0.3
done

# Results directory
RESULTS="res_PDEGQLET_MARLCoop/logs/results"

echo "[$(date)] RUNNING DBLoggerCustomStats/aggregated.overview.py" | tee -a QLSA-test-training-stats.log
python3 $RLLIB_SUMO_INFRASTRUCTURE/src/utils/DBLoggerCustomStats/aggregated.overview.py \
    --dir-tree $RESULTS --window 5 \
    --data "$AGGREGATED/aggregated-overview.json" \
    --graph "$AGGREGATED/aggregated-overview" # --profiler


echo "[$(date)] RUNNING DBLoggerCustomStats/mode.share.over.episodes.py" | tee -a QLSA-test-training-stats.log
python3 $RLLIB_SUMO_INFRASTRUCTURE/src/utils/DBLoggerCustomStats/mode.share.over.episodes.py \
    --dir-tree $RESULTS \
    --data "$AGGREGATED/mode-share-overview.json" \
    --graph "$AGGREGATED/mode-share-overview" # --profiler

echo "[$(date)] RUNNING DBLoggerCustomStats/aggregated.agents.decisions.py" | tee -a QLSA-test-training-stats.log
python3 $RLLIB_SUMO_INFRASTRUCTURE/src/utils/DBLoggerCustomStats/aggregated.agents.decisions.py \
    --dir-tree $RESULTS \
    --data "$AGGREGATED/agents-decisions-overview.json" \
    --graph "$AGGREGATED/agents-decisions-overview" # --profiler

echo "[$(date)] RUNNING DBLoggerCustomStats/agent.overview.py" | tee -a QLSA-test-training-stats.log
python3 $RLLIB_SUMO_INFRASTRUCTURE/src/utils/DBLoggerCustomStats/agent.overview.py \
    --dir-tree $RESULTS \
    --data "$AGENTS/agents-overview.json" \
    --graph "$AGENTS/overview" # --profiler

echo "[$(date)] RUNNING DBLoggerCustomStats/ett.insight.py" | tee -a QLSA-test-training-stats.log
python3 $RLLIB_SUMO_INFRASTRUCTURE/src/utils/DBLoggerCustomStats/ett.insight.py \
    --dir-tree $RESULTS \
    --graph "$ETT/ett" \
    --last-run

echo "[$(date)] RUNNING DBLoggerCustomStats/policy.insight.py" | tee -a QLSA-test-training-stats.log
python3 $RLLIB_SUMO_INFRASTRUCTURE/src/utils/DBLoggerCustomStats/policy.insight.py \
    --dir-tree $RESULTS --graph "$POLICIES/policy" --last-run

echo "[$(date)] RUNNING DBLoggerCustomStats/qvalues.insight.py" | tee -a QLSA-test-training-stats.log
python3 $RLLIB_SUMO_INFRASTRUCTURE/src/utils/DBLoggerCustomStats/qvalues.insight.py \
    --dir-tree $RESULTS --graph "$QVALUES/qvalues" --last-run

echo "[$(date)] RUNNING DBLoggerCustomStats/qvalues.insight.py [evolution]" | tee -a QLSA-test-training-stats.log
python3 $RLLIB_SUMO_INFRASTRUCTURE/src/utils/DBLoggerCustomStats/qvalues.insight.py \
    --dir-tree $RESULTS --graph "$QVALUESEVOL/qval.evol" --agent "$AGENT"

################################################################################

# It requires librsvg2-bin installed
for fname in $(find "$GRAPHS/" -name "*.svg")
do
	newname=$(echo $fname | sed -e 's/svg$/png/')
	rsvg-convert $fname -o $newname
done

python3 $RLLIB_SUMO_INFRASTRUCTURE/src/utils/HTMLGraphs/gallery.generator.py \
    --dir-tree "$GRAPHS/" --exp "Test Gallery"

################################################################################

echo "[$(date)] Cleanup." | tee -a QLSA-test-training-stats.log
rm -rv res_* sumo | tee -a QLSA-test-training-stats.log

echo "[$(date)] Done." | tee -a QLSA-test-training-stats.log
