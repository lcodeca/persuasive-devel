#!/bin/bash
set -e # error
set -u # unset variables

export SUMO_HOME="/home/alice/sumo"
export PATH="$SUMO_HOME/bin:$PATH"

export RLLIB_SUMO_INFRASTRUCTURE="/home/alice/devel/persuasive-devel"
export DISPLAY=:0

# Folder for the sumo simulation
SIMULATION="sumo"
mkdir -p $SIMULATION

echo "Experiment: TEST" > iteration.log

python3 $RLLIB_SUMO_INFRASTRUCTURE/src/runtesting.py \
    --algo QLSA --env LateMARL \
    --config "configuration.json" \
    --dir "test-cp-40" --target-checkpoint "res/checkpoints/checkpoint_40" \
    --testing-episodes 10 # --profiler
