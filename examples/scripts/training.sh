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

for i in {1..1}
do
    echo "[$(date)] RUNNING Iteration #$i" | tee -a iteration.log
    python3 $RLLIB_SUMO_INFRASTRUCTURE/src/runtraining.py \
        --algo PDEGQLET --env LateMARL \
        --epsilon 0.1 --action-distr 0.8 0.1 0.1 \
        --dir "res" --config "configuration.json" \
        --checkout-steps 10 --training-steps 1 \
        --alpha 0.1 --gamma 0.9 --decay 0.3 # --profiler
done
