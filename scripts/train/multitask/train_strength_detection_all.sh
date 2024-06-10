#!/bin/bash

# WA
## No Motion
### In-topic
bash scripts/train/multitask/train_strength_detection.sh False data/processed/arg_quality_rank_30k_wa.csv data/processed/ukp_sentential_argument_mining.csv 14.8 1 -no_motion-strength_detection_wa_in_topic ${1:-42}

### Cross-topic
bash scripts/train/multitask/train_strength_detection.sh False data/processed/arg_quality_rank_30k_cross_topic_wa.csv data/processed/ukp_sentential_argument_mining_cross_topic.csv 18.2 1 -no_motion-strength_detection_wa_cross_topic ${1:-42}

## Motion
### In-topic
bash scripts/train/multitask/train_strength_detection.sh True data/processed/arg_quality_rank_30k_wa.csv data/processed/ukp_sentential_argument_mining.csv 14.8 1 -motion-strength_detection_wa_in_topic ${1:-42}

### Cross-topic
bash scripts/train/multitask/train_strength_detection.sh True data/processed/arg_quality_rank_30k_cross_topic_wa.csv data/processed/ukp_sentential_argument_mining_cross_topic.csv 18.2 1 -motion-strength_detection_wa_cross_topic ${1:-42}
