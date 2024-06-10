#!/bin/bash

# AQ
bash scripts/run_argument_strength.sh 200
bash scripts/run_argument_strength.sh 201
bash scripts/run_argument_strength.sh 202

# ED
bash scripts/run_evidence_detection.sh 200
bash scripts/run_evidence_detection.sh 201
bash scripts/run_evidence_detection.sh 202

# AId
bash scripts/run_argument_identification.sh 200
bash scripts/run_argument_identification.sh 201
bash scripts/run_argument_identification.sh 202

# Multi-Task
bash scripts/train/multitask/train_all.sh 200
bash scripts/train/multitask/train_all.sh 201
bash scripts/train/multitask/train_all.sh 202

# Average
python3 src/results/average_seeds.py results/200 results/201 results/202 results/average
