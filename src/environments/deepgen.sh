#!/bin/bash
set -u
set -e

modes=$1
agents=$2
arrival=$3

python3 deepsumoagentgenerator.py \
	--net /home/codeca/Workspace/Alice/devel/scenarios/random-grid-sumo/scenario/random.net.xml \
	--shapes /home/codeca/Workspace/Alice/devel/scenarios/random-grid-sumo/scenario/poly.add.xml \
	--target 'taz' --destination 2190.42 1797.96 \
	--random-seed 987654321 --random-start 25200 "$arrival" \
	--default default_"$modes"_agent.json --num "$agents" \
	--out deep_agents_configs/deep_gold-"$agents"ag-"$modes"-rnd-"$arrival".json

# --origin 485.72 2781.14
# --random-start 25200 28800
# --random-start 25200 32400
