#!/bin/bash
set -u
set -e

# --origin 485.72 2781.14
# --random-start 25200 28800
# --random-start 25200 32400

bash deepgen.sh 5m 10 28800 rnd-orig-eval
bash deepgen.sh 5m 100 28800 rnd-orig-eval
bash deepgen.sh 5m 500 28800 rnd-orig-eval
bash deepgen.sh 5m 1000 28800 rnd-orig-eval

bash deepgen.sh 5m 10 32000 rnd-orig
bash deepgen.sh 5m 100 32000 rnd-orig
bash deepgen.sh 5m 500 32000 rnd-orig
bash deepgen.sh 5m 1000 32000 rnd-orig

bash deepgen-single.sh 5m 10 28800 single-orig-eval
bash deepgen-single.sh 5m 100 28800 single-orig-eval
bash deepgen-single.sh 5m 500 28800 single-orig-eval
bash deepgen-single.sh 5m 1000 28800 single-orig-eval

bash deepgen-single.sh 5m 10 32000 single-orig
bash deepgen-single.sh 5m 100 32000 single-orig
bash deepgen-single.sh 5m 500 32000 single-orig
bash deepgen-single.sh 5m 1000 32000 single-orig
