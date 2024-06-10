#!/bin/bash

python3 src/results/average_runs.py \
 "results/${2}/train/ARGUMENT_DETECTION_TASK/${1}/argument_detection_ukp-${1}-cross_topic_*_dev_*_test.json" \
 "results/${2}/train/ARGUMENT_DETECTION_TASK/${1}/argument_detection_ukp-${1}-cross_topic_average.json"