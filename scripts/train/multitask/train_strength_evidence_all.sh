#!/bin/bash

# WA
## No Motion
### In-topic
bash scripts/train/multitask/train_strength_evidence.sh False data/processed/arg_quality_rank_30k_wa.csv data/processed/evidences_sentences.csv 1.4 1 -no_motion-strength_evidence_wa_in_topic ${1:-42}

### Cross-topic
bash scripts/train/multitask/train_strength_evidence.sh False data/processed/arg_quality_rank_30k_cross_topic_wa.csv data/processed/evidences_sentences_cross_topic.csv 2.4 1 -no_motion-strength_evidence_wa_cross_topic ${1:-42}

## Motion
### In-topic
bash scripts/train/multitask/train_strength_evidence.sh True data/processed/arg_quality_rank_30k_wa.csv data/processed/evidences_sentences.csv 1.6 1 -motion-strength_evidence_wa_in_topic ${1:-42}

### Cross-topic
bash scripts/train/multitask/train_strength_evidence.sh True data/processed/arg_quality_rank_30k_cross_topic_wa.csv data/processed/evidences_sentences_cross_topic.csv 1.8 1 -motion-strength_evidence_wa_cross_topic ${1:-42}
