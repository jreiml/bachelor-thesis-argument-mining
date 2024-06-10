#!/bin/bash

# In-Topic
## No Motion
bash scripts/train/argument_detection_ukp/train.sh False data/processed/ukp_sentential_argument_mining.csv -no_motion-in_topic ${1:-42}
## Motion
bash scripts/train/argument_detection_ukp/train.sh True data/processed/ukp_sentential_argument_mining.csv -motion-in_topic ${1:-42}

# Cross-Topic
## No motion
bash scripts/train/argument_detection_ukp/train.sh False data/processed/ukp_sentential_argument_mining_cross_topic.csv -no_motion-cross_topic ${1:-42}
## Motion
bash scripts/train/argument_detection_ukp/train.sh True data/processed/ukp_sentential_argument_mining_cross_topic.csv -motion-cross_topic ${1:-42}