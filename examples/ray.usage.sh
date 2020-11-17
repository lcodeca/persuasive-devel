#! /bin/bash

#### ALICE ####
ray start --head --num-cpus 7 --num-gpus 0 --code-search-path=/home/alice/devel/persuasive-devel/

#### FASTIDIO ####
ray start --address='192.168.1.25:6379' --redis-password='5241590000000000' --num-cpus 1 --num-gpus 1 --code-search-path=/home/alice/devel/persuasive-devel/

#### Both ####
htop -s PERCENT_MEM
