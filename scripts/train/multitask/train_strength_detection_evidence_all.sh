#!/bin/bash

# WA
## No Motion
### In-topic
bash scripts/train/multitask/train_strength_detection_evidence.sh False data/processed/arg_quality_rank_30k_wa.csv data/processed/ukp_sentential_argument_mining.csv data/processed/evidences_sentences.csv 14.8 1 10.5 -no_motion-strength_detection_evidence_wa_in_topic ${1:-42}

### Cross-topic
bash scripts/train/multitask/train_strength_detection_evidence.sh False data/processed/arg_quality_rank_30k_cross_topic_wa.csv data/processed/ukp_sentential_argument_mining_cross_topic.csv data/processed/evidences_sentences_cross_topic.csv 18.2 1 7.6 -no_motion-strength_detection_evidence_wa_cross_topic ${1:-42}

## Motion
### In-topic
bash scripts/train/multitask/train_strength_detection_evidence.sh True data/processed/arg_quality_rank_30k_wa.csv data/processed/ukp_sentential_argument_mining.csv data/processed/evidences_sentences.csv 14.8 1 9.25 -motion-strength_detection_evidence_wa_in_topic ${1:-42}

### Cross-topic
bash scripts/train/multitask/train_strength_detection_evidence.sh True data/processed/arg_quality_rank_30k_cross_topic_wa.csv data/processed/ukp_sentential_argument_mining_cross_topic.csv data/processed/evidences_sentences_cross_topic.csv 18.2 1 10 -motion-strength_detection_evidence_wa_cross_topic ${1:-42}
