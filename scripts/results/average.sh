#!/bin/bash

python3 src/results/average_runs.py \
 "results/${3}/${1}/${2}/no_motion/*.json" \
 "results/${3}/${1}/${2}/average_no_motion.json"

python3 src/results/average_runs.py \
 "results/${3}/${1}/${2}/motion/*.json" \
 "results/${3}/${1}/${2}/average_motion.json"

python3 src/results/average_runs.py \
 "results/${3}/${1}/${2}/**/*in_topic*.json" \
 "results/${3}/${1}/${2}/average_in_topic.json"

 python3 src/results/average_runs.py \
 "results/${3}/${1}/${2}/**/*cross_topic*.json" \
 "results/${3}/${1}/${2}/average_cross_topic.json"

python3 src/results/average_runs.py \
 "results/${3}/${1}/${2}/**/*.json" \
 "results/${3}/${1}/${2}/average.json"