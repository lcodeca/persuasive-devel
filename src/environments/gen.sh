#!/bin/bash

modes=$1
agents=$2

python3 sumoagentgenerator.py \
	--net /home/codeca/Workspace/Alice/devel/scenarios/random-grid-sumo/scenario/random.net.xml \
	--from-edge -6040 --to-edge 2370 --random-seed 987654321 --random-start 25200 28800 \
	--default default_"$modes"_agent.json --num $agents --out gold-"$agents"ag-"$modes"-rndstart.json
