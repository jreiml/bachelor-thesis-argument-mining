#!/bin/bash

# No Motion
## In-topic
bash scripts/train/multitask/train_detection_evidence.sh False data/processed/ukp_sentential_argument_mining.csv data/processed/evidences_sentences.csv 1 10.5 -no_motion-detection_evidence_in_topic ${1:-42}

## Cross-topic
bash scripts/train/multitask/train_detection_evidence.sh False data/processed/ukp_sentential_argument_mining_cross_topic.csv data/processed/evidences_sentences_cross_topic.csv 1 7.6 -no_motion-detection_evidence_cross_topic ${1:-42}

# Motion
## In-topic
bash scripts/train/multitask/train_detection_evidence.sh True data/processed/ukp_sentential_argument_mining.csv data/processed/evidences_sentences.csv 1 9.25 -motion-detection_evidence_in_topic ${1:-42}

## Cross-topic
bash scripts/train/multitask/train_detection_evidence.sh True data/processed/ukp_sentential_argument_mining_cross_topic.csv data/processed/evidences_sentences_cross_topic.csv 1 10 -motion-detection_evidence_cross_topic ${1:-42}
