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

echo "Full TESTs for QLSA Training" > full-QLSA-test-training.log

################################################################################

echo "[$(date)] RUNNING QLSA - MARL" | tee -a full-QLSA-test-training.log
for i in {1..2} ;do
    echo "[$(date)] Iteration $i " | tee -a full-QLSA-test-training.log
    python3 -u $RLLIB_SUMO_INFRASTRUCTURE/src/runQLSAtraining.py \
        --algo QLSA --env MARL \
        --dir "res_QLSA_MARL" --config "configuration.json" \
        --checkout-steps 2 --training-steps 2 \
        --alpha 0.1 --gamma 0.9 --epsilon 0.0001
done

echo "[$(date)] RUNNING QLSA - LateMARL" | tee -a full-QLSA-test-training.log
for i in {1..2} ;do
    echo "[$(date)] Iteration $i " | tee -a full-QLSA-test-training.log
    python3 -u $RLLIB_SUMO_INFRASTRUCTURE/src/runQLSAtraining.py \
        --algo QLSA --env LateMARL \
        --dir "res_QLSA_LateMARL" --config "configuration.json" \
        --checkout-steps 2 --training-steps 2 \
        --alpha 0.1 --gamma 0.9 --epsilon 0.0001
done

echo "[$(date)] RUNNING QLSA - MARLCoop" | tee -a full-QLSA-test-training.log
for i in {1..2} ;do
    echo "[$(date)] Iteration $i " | tee -a full-QLSA-test-training.log
    python3 -u $RLLIB_SUMO_INFRASTRUCTURE/src/runQLSAtraining.py \
        --algo QLSA --env MARLCoop \
        --dir "res_QLSA_MARLCoop" --config "configuration.json" \
        --checkout-steps 2 --training-steps 2 \
        --alpha 0.1 --gamma 0.9 --epsilon 0.0001
done

# ################################################################################

echo "[$(date)] RUNNING QLET - MARL" | tee -a full-QLSA-test-training.log
for i in {1..2} ;do
    echo "[$(date)] Iteration $i " | tee -a full-QLSA-test-training.log
    python3 -u $RLLIB_SUMO_INFRASTRUCTURE/src/runQLSAtraining.py \
        --algo QLET --env MARL \
        --dir "res_QLET_MARL" --config "configuration.json" \
        --checkout-steps 2 --training-steps 2 \
        --alpha 0.1 --gamma 0.9 --epsilon 0.0001 --decay 0.3
done

echo "[$(date)] RUNNING QLET - LateMARL" | tee -a full-QLSA-test-training.log
for i in {1..2} ;do
    echo "[$(date)] Iteration $i " | tee -a full-QLSA-test-training.log
    python3 -u $RLLIB_SUMO_INFRASTRUCTURE/src/runQLSAtraining.py \
        --algo QLET --env LateMARL \
        --dir "res_QLET_LateMARL" --config "configuration.json" \
        --checkout-steps 2 --training-steps 2 \
        --alpha 0.1 --gamma 0.9 --epsilon 0.0001 --decay 0.3
done

echo "[$(date)] RUNNING QLET - MARLCoop" | tee -a full-QLSA-test-training.log
for i in {1..2} ;do
    echo "[$(date)] Iteration $i " | tee -a full-QLSA-test-training.log
    python3 -u $RLLIB_SUMO_INFRASTRUCTURE/src/runQLSAtraining.py \
        --algo QLET --env MARLCoop \
        --dir "res_QLET_MARLCoop" --config "configuration.json" \
        --checkout-steps 2 --training-steps 2 \
        --alpha 0.1 --gamma 0.9 --epsilon 0.0001 --decay 0.3
done

# ################################################################################

echo "[$(date)] RUNNING PDEGQLET - MARL" | tee -a full-QLSA-test-training.log
for i in {1..2} ;do
    echo "[$(date)] Iteration $i " | tee -a full-QLSA-test-training.log
    python3 -u $RLLIB_SUMO_INFRASTRUCTURE/src/runQLSAtraining.py \
        --algo PDEGQLET --env MARL \
        --epsilon 0.1 --action-distr 0.9 0.1 \
        --dir "res_PDEGQLET_MARL" --config "configuration.json" \
        --checkout-steps 2 --training-steps 2 \
        --alpha 0.1 --gamma 0.9 --decay 0.3
done

echo "[$(date)] RUNNING PDEGQLET - LateMARL" | tee -a full-QLSA-test-training.log
for i in {1..2} ;do
    echo "[$(date)] Iteration $i " | tee -a full-QLSA-test-training.log
    python3 -u $RLLIB_SUMO_INFRASTRUCTURE/src/runQLSAtraining.py \
        --algo PDEGQLET --env LateMARL \
        --epsilon 0.1 --action-distr 0.9 0.1 \
        --dir "res_PDEGQLET_LateMARL" --config "configuration.json" \
        --checkout-steps 2 --training-steps 2 \
        --alpha 0.1 --gamma 0.9 --decay 0.3
done

echo "[$(date)] RUNNING PDEGQLET - MARLCoop" | tee -a full-QLSA-test-training.log
for i in {1..2} ;do
    echo "[$(date)] Iteration $i " | tee -a full-QLSA-test-training.log
    python3 -u $RLLIB_SUMO_INFRASTRUCTURE/src/runQLSAtraining.py \
        --algo PDEGQLET --env MARLCoop \
        --epsilon 0.1 --action-distr 0.9 0.1 \
        --dir "res_PDEGQLET_MARLCoop" --config "configuration.json" \
        --checkout-steps 2 --training-steps 2 \
        --alpha 0.1 --gamma 0.9 --decay 0.3
done

################################################################################

echo "[$(date)] Cleanup." | tee -a full-QLSA-test-training.log
rm -rv res_* sumo | tee -a full-QLSA-test-training.log

echo "[$(date)] Done." | tee -a full-QLSA-test-training.log
