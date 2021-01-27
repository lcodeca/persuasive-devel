#! /bin/bash

#### ALICE ####
ray start --head --num-cpus 7 --num-gpus 0 --code-search-path=/home/alice/devel/persuasive-devel/

#### FASTIDIO ####
ray start --address='192.168.1.25:6379' --redis-password='5241590000000000' --num-cpus 1 --num-gpus 1 --code-search-path=/home/alice/devel/persuasive-devel/

#### Both ####
htop -s PERCENT_MEM

#### HOPPER - Ray master #### #64424509440 --port 0
ray start --head --num-cpus 20 --num-gpus 1 --code-search-path=/home/alice/devel/persuasive-devel/ --object-store-memory 64324509440

#### HOPPER - Others
ray start --address='172.17.0.2:6379' --redis-password='5241590000000000' --num-cpus 0 --num-gpus 0 --object-store-memory 10637418240
